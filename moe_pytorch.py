import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import copy
import json
import warnings
warnings.filterwarnings('ignore')

class SynergisticMoE(nn.Module):
    def __init__(self, global_input_dim, expert_dims):
        super(SynergisticMoE, self).__init__()
        self.num_experts = len(expert_dims)
        
        self.experts = nn.ModuleList([
            nn.Linear(dim, 1) for dim in expert_dims
        ])
        
        self.router = nn.Sequential(
            nn.Linear(global_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_experts),
            nn.Sigmoid()
        )
        
        self.synthesizer = nn.Sequential(
            nn.Linear(self.num_experts + 4, 16),  # 4 эксперта + 3 флага (S, N, Amino) + 1 флаг (Vapor Pressure)
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_global, x_parts):
        gates = self.router(x_global) # [batch, num_experts]
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            e_out = expert(x_parts[i])
            expert_outputs.append(e_out)
            
        expert_outputs = torch.cat(expert_outputs, dim=1) # [batch, num_experts]
        weighted_outputs = gates * expert_outputs
        
        # Последние 4 колонки в x_global — это 3 химических флага и 1 флаг Vapor Pressure
        x_flags = x_global[:, -4:]
        synth_input = torch.cat([weighted_outputs, x_flags], dim=1)
        final_output = self.synthesizer(synth_input)
        
        return final_output, gates

def main():
    print("Загрузка ПОЛНОГО датасета Klio + Vapor Pressure (1272 молекулы)")
    df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\klio_full_with_imputed_vp.csv")
    
    if 'Preferred_Imax' in df.columns:
        y_true = df['Preferred_Imax'].values
    else:
        y_true = df['Imax'].values
        
    drop_cols = ['SMILES', 'Imax', 'Preferred_Imax', 'name', 'cid', 'CID', 'Canonical_Name', 'intensity_class', 'class', 'label', 'target', 'cluster', 'CID_merge', 'vp_mmhg_pubchem', 'Vapor_Pressure', 'log10_Vapor_Pressure']
    X_raw = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    
    # === ГЕНЕРАЦИЯ ХИМИЧЕСКИХ ФЛАГОВ ===
    names = df['Canonical_Name'].str.lower()
    
    # 1. Серные соединения
    s_keys = ['sulfid', 'mercapt', 'thio', 'thian', 'thiazol', 'thiophen', 'sulfox', 'sulfon']
    X_raw['Has_Sulfur'] = names.apply(lambda x: 1.0 if any(k in str(x) for k in s_keys) else 0.0)
    
    # 2. Азотсодержащие кольца
    n_keys = ['pyrazin', 'pyridin', 'quinolin', 'triazol', 'pyrrol', 'lutidin', 'indol', 'imidazol']
    X_raw['Has_Nitrogen_Ring'] = names.apply(lambda x: 1.0 if any(k in str(x) for k in n_keys) else 0.0)
    
    # 3. Аминокислоты
    a_keys = ['alanine', 'leucine', 'glutamic', 'glutamine', 'glycine', 'valine', 'proline', 'serine', 'threonine', 'cysteine', 'methionine', 'aspartic', 'asparagine', 'histidine', 'phenylalanine', 'tyrosine', 'tryptophan', 'arginine', 'lysine']
    X_raw['Is_Amino_Acid'] = names.apply(lambda x: 1.0 if any(k in str(x) for k in a_keys) else 0.0)
    
    # === ДОБАВЛЕНИЕ VAPOR PRESSURE ===
    X_raw['VP_Flag'] = df['log10_Vapor_Pressure'].fillna(df['log10_Vapor_Pressure'].median())
    
    X_all = X_raw.values
    
    with open(r"C:\Users\igork\Desktop\main\Project\moe_selected_features.json", 'r') as f:
        selected_features = json.load(f)
        
    expert_feature_lists = []
    for group in ['Steric', 'Electrostatic', 'Hydrophobic', 'H_Bonds']:
        if group in selected_features and len(selected_features[group]) > 0:
            expert_feature_lists.append(selected_features[group])
            
    n_folds = 20
    rs = ShuffleSplit(n_splits=n_folds, test_size=0.1, random_state=42)
    
    test_rmses = []
    test_r2s = []
    all_oof_predictions = np.zeros(len(y_true))
    prediction_counts = np.zeros(len(y_true))
    
    print(f"\nНачинаем {n_folds}-кратный прогон (Train: ~80%, Val: ~10%, Test: 10%)...")
    
    for fold, (train_val_idx, test_idx) in enumerate(rs.split(X_all)):
        X_train_val, y_train_val = X_all[train_val_idx], y_true[train_val_idx]
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.11, random_state=fold)
        y_test = y_true[test_idx]
        
        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_all[test_idx])
        
        # Helper to extract parts
        def get_parts(X, features):
            parts = []
            for feat_list in features:
                indices = [df.columns.get_loc(c) for c in feat_list if c in df.columns]
                parts.append(torch.FloatTensor(X[:, indices]))
            return parts

        X_parts_tr = get_parts(X_train_s, expert_feature_lists)
        X_parts_val = get_parts(X_val_s, expert_feature_lists)
        X_parts_te = get_parts(X_test_s, expert_feature_lists)
        
        t_X_tr = torch.FloatTensor(X_train_s)
        t_y_tr = torch.FloatTensor(y_train).view(-1, 1)
        t_X_val = torch.FloatTensor(X_val_s)
        t_y_val = torch.FloatTensor(y_val).view(-1, 1)
        t_X_te = torch.FloatTensor(X_test_s)
        
        expert_dims = [p.shape[1] for p in X_parts_tr]
        model = SynergisticMoE(global_input_dim=X_all.shape[1], expert_dims=expert_dims)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)
        
        best_val_loss = float('inf')
        patience, patience_counter = 150, 0
        best_weights = None
        
        for epoch in range(1500):
            model.train()
            optimizer.zero_grad()
            preds, _ = model(t_X_tr, X_parts_tr)
            loss = criterion(preds, t_y_tr)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                val_preds, _ = model(t_X_val, X_parts_val)
                val_loss = criterion(val_preds, t_y_val).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience: break
                
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            test_preds, _ = model(t_X_te, X_parts_te)
            test_preds = test_preds.cpu().numpy().flatten()
            
            rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            if np.var(y_test) == 0:
                r2 = 0.0
            else:
                r2 = r2_score(y_test, test_preds)
            
            test_rmses.append(rmse)
            test_r2s.append(r2)
            
            all_oof_predictions[test_idx] += test_preds.flatten()
            prediction_counts[test_idx] += 1
            
        print(f"Эксперимент {fold+1:02d}: Невиданные молекулы ({len(test_idx)} шт) -> RMSE = {rmse:.2f}, R2 = {r2:.3f}")

    print("\n================== ИТОГОВЫЕ РЕЗУЛЬТАТЫ СИНТЕЗАТОРА ==================")
    print(f"Средний RMSE на 20 слепых тестах: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f}")
    print(f"Средний R2 на 20 слепых тестах:   {np.mean(test_r2s):.3f} ± {np.std(test_r2s):.3f}")
    print("=====================================================================")
    
    # Сохраняем агрегированные предсказания
    df['MoE_Synergy_Pred'] = np.divide(all_oof_predictions, prediction_counts, out=np.zeros_like(all_oof_predictions), where=prediction_counts!=0)
    df.to_csv(r"C:\Users\igork\Desktop\main\Project\moe_synergy_results.csv", index=False)

if __name__ == "__main__":
    main()
