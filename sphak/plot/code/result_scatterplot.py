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

# --- Load data ---
plant_df = pd.read_csv('/content/plant_result.csv')
animal_df = pd.read_csv('/content/animal_result.csv')

plant_df['Domain'] = 'Plant'
animal_df['Domain'] = 'Animal'
df = pd.concat([animal_df, plant_df], ignore_index=True)
df = df[df['Family'].notna() & df['Best_Family'].notna()]
df['Correct'] = df['Family'] == df['Best_Family']
df['Panel'] = df['Correct'].map({True: 'Correctly Predicted', False: 'Not Correctly Predicted'})
df['Group'] = df['Domain']
df['Family_Label'] = df['Family']
df['SP_Score'] = df['Prediction_Score']

panel_order = ['Correctly Predicted', 'Not Correctly Predicted']

# --- Define custom colormap ---
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('custom_byr', [
    (0.00, '#000000'),  # Black (0 coverage - very distinct)
    (0.005, '#0000ff'),  # Bright Blue
    (0.01, '#4b0082'),  # Indigo / Purple
    (0.06, '#8000ff'),  # Violet
    (0.1, '#da70d6'),  # Orchid (light purple)
    (0.2, '#ffff66'),  # Soft Yellow
    (0.3, '#ffff00'),  # Yellow
    (0.4, '#ffcc00'),  # Golden Yellow
    (0.5, '#ff9933'),  # Orange
    (0.6, '#ff6600'),  # Deep Orange
    (0.7, '#ff3300'),  # Reddish Orange
    (0.8, '#ff0000'),  # Red
    (0.9, '#cc0000'),  # Dark Red
    (1.00, '#800000')   # Maroon / Deep Red
])




# --- Fixed dot size ---
fixed_size = 40

# --- Threshold to coverage ---
coverage_threshold = -1

# --- Function to create plot for each domain ---
def create_domain_plot(domain_name, output_file):
    domain_df = df[df['Domain'] == domain_name].copy()

    # Order families for consistent y-axis
    families = sorted(domain_df['Family_Label'].unique())
    family_to_y = {fam: i for i, fam in enumerate(families)}
    domain_df['y_pos'] = domain_df['Family_Label'].map(family_to_y)

    # Filter out zero or near-zero coverage points
    domain_df = domain_df[domain_df['Coverage'] > coverage_threshold]

    # Normalize coverage for coloring
    coverage_norm = mpl.colors.Normalize(vmin=domain_df['Coverage'].min(), vmax=domain_df['Coverage'].max())

    fig, axes = plt.subplots(1, 2, figsize=(10, len(families) * 0.3 + 2), sharey=True)

    for ax, panel in zip(axes, panel_order):
        subdf = domain_df[domain_df['Panel'] == panel]

        # Add vertical dotted line at 0.5 only on "Not Correctly Predicted"
        if panel == 'Not Correctly Predicted':
          ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)


        facecolors = cmap(coverage_norm(subdf['Coverage']))
        edgecolors = cmap(coverage_norm(subdf['Coverage'] * 0.85))

        # Plot coverage points (only > threshold)
        ax.scatter(
            subdf['SP_Score'], subdf['y_pos'],
            c=facecolors,
            edgecolors=edgecolors,
            linewidths=0.5,
            s=fixed_size,
            alpha=0.9,
            zorder=3
        )

        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        ax.set_xlabel('SP Score')
        ax.set_title(panel, pad=10)
        ax.set_yticks(np.arange(len(families)))
        ax.set_yticklabels(families)
        ax.yaxis.grid(True, linestyle=':', color='gray', linewidth=0.5)


    # --- Colorbar for coverage ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=coverage_norm)
    sm.set_array([])
    cax = inset_axes(axes[-1], width="10%", height="80%", loc='center left',
                     bbox_to_anchor=(1.10, 0., 1, 1), bbox_transform=axes[-1].transAxes, borderpad=0)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('k-mer Coverage')
    cbar.ax.tick_params()

    plt.tight_layout(rect=[0, 0, 0.95, 1.0])
    plt.savefig(output_file, dpi=900, bbox_inches='tight')
    plt.show()

# --- Create plots ---
create_domain_plot("Animal", "animal_plot.png")
create_domain_plot("Plant", "plant_plot.png")
