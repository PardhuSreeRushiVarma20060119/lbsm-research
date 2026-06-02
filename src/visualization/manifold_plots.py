import numpy as np
import pandas as pd
import plotly.express as px

X = np.load("../../data/raw/nb02/X_umap3.npy")
y = np.load("../../data/raw/nb02/y_labels.npy")

print("Unique labels:", np.unique(y, return_counts=True))

names = {
    0: "Stable",
    1: "Exploratory",
    2: "Adaptive",
    3: "Unstable"
}

df = pd.DataFrame({
    "UMAP1": X[:,0],
    "UMAP2": X[:,1],
    "UMAP3": X[:,2],
    "Regime": [names[i] for i in y]
})

fig = px.scatter_3d(
    df,
    x="UMAP1",
    y="UMAP2",
    z="UMAP3",
    color="Regime",
    color_discrete_map={
        "Stable": "#2ecc71",       # green
        "Exploratory": "#3498db",  # blue
        "Adaptive": "#9b59b6",     # purple
        "Unstable": "#e74c3c"      # red
    }
)

fig.update_traces(
    marker=dict(
        size=0.8,
        opacity=0.20
    )
)

fig.update_layout(
    template="plotly_dark",
    scene=dict(
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        zaxis_title="UMAP3",
        aspectmode="cube",
        bgcolor="black"
    ),
    paper_bgcolor="black",
    legend_title_text="Regime"
)

fig.write_html("lbsm_umap3d.html")
print("saved")