import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib import font_manager as fm

# === Suppress only font-related warnings ===
warnings.filterwarnings("ignore", message="findfont: Font family.*not found")

# === Check for font file and load Arial or fallback to DejaVu Sans ===
font_path = "arial.ttf"

if os.path.exists(font_path):
    print(f"✅ Font file '{font_path}' found.")
else:
    print(f"❌ Font file '{font_path}' NOT found. Please check the path.")

try:
    # Load custom font and register with Matplotlib
    custom_font = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = custom_font.get_name()
    print(f"✅ Successfully loaded font: {custom_font.get_name()}")
except (RuntimeError, OSError) as e:
    print(f"❌ Could not load font from '{font_path}'. Using fallback: DejaVu Sans")
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["font.size"] = 8
print(f"📊 Final font used: {plt.rcParams['font.family']}")

# === Load dataset ===
animal = pd.read_csv("animal_out_of_sample_result.csv")
animal["Coverage_pct"] = animal["Coverage"] * 100

# === Split data by prediction ===
binds = animal[animal["Prediction"] == 1]
not_binds = animal[animal["Prediction"] == 0]

# === Simplified virus labels for annotation ===
#simplified_labels = {
#    "Tomato leaf curl New Delhi virus": "ToLCNDV",
#    "Chilli leaf curl virus": "ChiLCV"
#}

simplified_labels = {
    "Murine hepatitis virus" : 'MHV',
    "Canine respiratory coronavirus": 'CRCoV '
}

# === Create scatter plot ===
plt.figure(figsize=(8, 5))  # Increased width for more label space

# Plot negative and positive predictions
plt.scatter(not_binds["Coverage_pct"], not_binds["Prediction_Score"],
            color='black', s=20, label="Host Negative")
plt.scatter(binds["Coverage_pct"], binds["Prediction_Score"],
            color='red', s=20, label="Host Positive")

# Axis labels
plt.xlabel("k-mer Coverage of Query Sequence (%)")
plt.ylabel("Spillover Prediction Score (SP Score)")

# Add grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

# === Annotate key viruses ===
texts = []
for virus_name, simple_name in simplified_labels.items():
    matches = animal[animal["Virus"].str.contains(virus_name, case=False, na=False)]
    if not matches.empty:
        match = matches.loc[matches["Coverage_pct"].idxmax()]
        x = match["Coverage_pct"]
        y = match["Prediction_Score"]

        plt.scatter(x, y, color='#00b33c', marker='*', s=50, zorder=5)
        texts.append(plt.text(x, y, simple_name, fontsize=7))

# Adjust overlapping text annotations
adjust_text(
    texts,
    ax=plt.gca(),
    expand_points=(2, 2),
    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
    force_text=(0.5, 0.5),
    lim=100
)

# === Add legend inside the plot ===
plt.legend(loc='upper left', fontsize=8, title="Prediction", title_fontsize=8, frameon=True)


# Final layout and save
plt.tight_layout()
plt.subplots_adjust(top=.9)  # Extra space at top for labels

# Save high-quality versions
plt.savefig('scatterplot_plant.png', dpi=1200, bbox_inches='tight')
plt.savefig('scatterplot_plant.pdf', bbox_inches='tight')

plt.show()

