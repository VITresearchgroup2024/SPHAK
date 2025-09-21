import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm

# --- Suppress only font-related warnings ---
warnings.filterwarnings("ignore", message="findfont: Font family.*not found")

# --- Check for Arial font file and set font ---
font_path = "arial.ttf"
if os.path.exists(font_path):
    try:
        arial_font = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = arial_font.get_name()
        print(f"✅ Using custom font: {arial_font.get_name()}")
    except Exception as e:
        print(f"❌ Could not load Arial: {e}")
        plt.rcParams["font.family"] = "DejaVu Sans"
else:
    print("❌ Arial font file not found. Using DejaVu Sans.")
    plt.rcParams["font.family"] = "DejaVu Sans"

# --- Set global font size ---
plt.rcParams["font.size"] = 10

# --- Load influenza data ---
animal_df = pd.read_csv('/content/influenza_results.csv')

# Assign domain
animal_df['Domain'] = 'Animal'

# Filter incomplete entries
df = animal_df[animal_df['Family'].notna() & animal_df['Best_Family'].notna()].copy()

# Add columns for correctness, panel grouping, etc.
df['Correct'] = df['Family'] == df['Best_Family']
df['Panel'] = df['Correct'].map({True: 'Correctly Predicted', False: 'Not Correctly Predicted'})
df['Group'] = df['Domain']
df['Family_Label'] = df['Family']
df['SP_Score'] = df['Prediction_Score']

panel_order = ['Correctly Predicted', 'Not Correctly Predicted']

# --- Define custom colormap ---
cmap = LinearSegmentedColormap.from_list('custom_byr', [
    (0.00, '#000000'),
    (0.005, '#0000ff'),
    (0.01, '#4b0082'),
    (0.06, '#8000ff'),
    (0.1, '#da70d6'),
    (0.2, '#ffff66'),
    (0.3, '#ffff00'),
    (0.4, '#ffcc00'),
    (0.5, '#ff9933'),
    (0.6, '#ff6600'),
    (0.7, '#ff3300'),
    (0.8, '#ff0000'),
    (0.9, '#cc0000'),
    (1.00, '#800000')
])

# === Simplified virus labels for annotation ===
simplified_labels = {
    "Influenza A virus (A/swine/MO/15534/2010(H1N1))": "H1N1",
    "Influenza A virus (A/swine/Taiwan/NPUST0009/2013(H3N1))": "H3N1",
    "Influenza D virus (D/bovine/Kansas/13-21/2012)": "influenza D"
}

df['Virus_Label'] = df['Virus'].map(simplified_labels).fillna("")

# Pick only one row per Virus (highest Prediction_Score) for annotation
df_for_labels = (
    df.sort_values("Prediction_Score", ascending=False)
      .drop_duplicates(subset=["Virus"], keep="first")
)

# Fixed size for dots
fixed_size = 40

# Threshold for coverage filtering
coverage_threshold = -1

# --- Function to create plot ---
def create_domain_plot(domain_name, output_file):
    domain_df = df[df['Domain'] == domain_name].copy()
    families = sorted(domain_df['Family_Label'].unique())
    family_to_y = {fam: i for i, fam in enumerate(families)}
    domain_df['y_pos'] = domain_df['Family_Label'].map(family_to_y)

    domain_df = domain_df[domain_df['Coverage'] > coverage_threshold]
    coverage_norm = mpl.colors.Normalize(vmin=domain_df['Coverage'].min(),
                                         vmax=domain_df['Coverage'].max())

    fig, axes = plt.subplots(1, 2, figsize=(10, len(families) * 0.3 + 2), sharey=True)

    for ax, panel in zip(axes, panel_order):
        subdf = domain_df[domain_df['Panel'] == panel]

        if panel == 'Not Correctly Predicted':
            ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)

        facecolors = cmap(coverage_norm(subdf['Coverage']))
        edgecolors = cmap(coverage_norm(subdf['Coverage'] * 0.85))

        ax.scatter(
            subdf['SP_Score'], subdf['y_pos'],
            c=facecolors,
            edgecolors=edgecolors,
            linewidths=0.5,
            s=fixed_size,
            alpha=0.9,
            zorder=3
        )

        # --- Annotate simplified virus labels (only best-scoring row per Virus) ---
        subdf_labels = df_for_labels[df_for_labels['Panel'] == panel]
        subdf_labels = subdf_labels[subdf_labels['Domain'] == domain_name]

        for _, row in subdf_labels.iterrows():
            if row['Virus_Label'] != "":
                ax.text(
                    row['SP_Score'],
                    family_to_y[row['Family_Label']] + 0.01,
                    row['Virus_Label'],
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    rotation=90,
                    color='black'
                )

        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        ax.set_xlabel('SP Score')
        ax.set_title(panel, pad=10)
        ax.set_yticks(np.arange(len(families)))
        ax.set_yticklabels(families)

        if ax == axes[0]:
            ax.set_ylabel(f"{domain_name} Virus Family", fontsize=8, labelpad=20, rotation=90, loc='center')

        ax.yaxis.grid(True, linestyle=':', color='gray', linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=coverage_norm)
    sm.set_array([])
    cax = inset_axes(axes[-1], width="10%", height="80%", loc='center left',
                     bbox_to_anchor=(1.10, 0., 1, 1), bbox_transform=axes[-1].transAxes, borderpad=0)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('k-mer Coverage')
    cbar.ax.tick_params()

    plt.tight_layout(rect=[0, 0, 0.95, 1.0])
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')
    plt.show()

# --- Generate plot only for influenza ---
create_domain_plot("Animal", "influenza_plot.png")
