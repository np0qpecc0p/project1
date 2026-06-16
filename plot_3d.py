import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import os
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\igork\Desktop\main\ai-program-2026\project_1\data\waka_dragon_merged.csv")
with open(r"C:\Users\igork\Desktop\main\Project\moe_selected_features.json", 'r') as f:
    features = json.load(f)

output_dir = r"C:\Users\igork\Desktop\main\Project\plots_3d"
os.makedirs(output_dir, exist_ok=True)

y_true = df['Imax'].values
scaler = StandardScaler()
thresholds = {'Steric': 0.25, 'Electrostatic': 0.30, 'Hydrophobic': 0.25, 'H_Bonds': 0.25}

hover_name = df['cid'] if 'cid' in df.columns else df.index

for group, feats in features.items():
    if len(feats) < 3: continue
    
    top_3 = feats[:3]
    
    # Получаем инлаеров жестким отбором
    X = df[feats].fillna(0).values
    X_scaled = scaler.fit_transform(X)
    thresh = np.std(y_true) * thresholds[group]
    ransac = RANSACRegressor(estimator=SVR(kernel='rbf', C=20.0, epsilon=0.05), min_samples=0.1, residual_threshold=thresh, max_trials=200, random_state=42)
    ransac.fit(X_scaled, y_true)
    inliers = ransac.inlier_mask_
    
    # Разделяем датафрейм
    df_in = df[inliers]
    df_out = df[~inliers]
    
    fig = go.Figure()
    
    # Сначала рисуем Аутлаеров (Шум, серые, мелкие)
    fig.add_trace(go.Scatter3d(
        x=df_out[top_3[0]], y=df_out[top_3[1]], z=df_out[top_3[2]],
        mode='markers',
        marker=dict(size=3, color='rgba(100, 100, 100, 0.2)', line=dict(width=0)),
        name='Шум (Аутлаеры)',
        text=hover_name[~inliers],
        hoverinfo='text'
    ))
    
    # Теперь рисуем Инлаеров (Чистые специалисты, цветные, крупные)
    fig.add_trace(go.Scatter3d(
        x=df_in[top_3[0]], y=df_in[top_3[1]], z=df_in[top_3[2]],
        mode='markers',
        marker=dict(
            size=7,
            color=df_in['Imax'],
            colorscale='Turbo',
            showscale=True,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name='Идеальные Инлаеры',
        text=hover_name[inliers],
        hovertemplate='CID: %{text}<br>Imax: %{marker.color:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title=f"3D кластер: {group} (Только инлаеры цветные)",
        scene=dict(
            xaxis_title=top_3[0],
            yaxis_title=top_3[1],
            zaxis_title=top_3[2]
        )
    )
    
    out_path = os.path.join(output_dir, f"{group}_3d.html")
    fig.write_html(out_path)
    print(f"Готово -> {out_path}")
