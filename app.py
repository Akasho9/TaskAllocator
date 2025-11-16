"""
Visualization Dashboard for GNN Task Allocation System.

Authors: Group-27
    - Nikita Dhiman (M25AI1001)(G24AIT001)
    - Hensi Solanki (M25AI1090)(G24AIT115)
    - Akash Singh (M25AI1043)(G24AIT054)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import networkx as nx
from matplotlib.patches import Patch

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)

print("Creating visualizations...")
print("=" * 70)

# Load data
developers_df = pd.read_csv('data/developers.csv')
tasks_df = pd.read_csv('data/tasks.csv')
technologies_df = pd.read_csv('data/technologies.csv')

# Load latest allocation report
results_dir = Path('results')
latest_report = sorted(results_dir.glob('allocation_report_gnn_*.csv'))[-1]
allocation_df = pd.read_csv(latest_report)

print(f"Loaded data from: {latest_report.name}\n")

# ============================================================================
# 1. TASK ALLOCATION DISTRIBUTION
# ============================================================================
print("1. Creating Task Allocation Distribution chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Developer workload
dev_counts = allocation_df['Developer Name'].value_counts()
colors = plt.cm.viridis(np.linspace(0, 1, len(dev_counts)))

ax1.barh(dev_counts.index, dev_counts.values, color=colors)
ax1.set_xlabel('Number of Tasks Allocated', fontsize=11, fontweight='bold')
ax1.set_ylabel('Developer', fontsize=11, fontweight='bold')
ax1.set_title('Workload Distribution Across Developers', fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)

for i, (name, count) in enumerate(dev_counts.items()):
    ax1.text(count + 0.1, i, str(count), va='center', fontweight='bold')

# Priority distribution
priority_counts = allocation_df['Priority'].value_counts()
priority_order = ['Critical', 'High', 'Medium', 'Low']
priority_counts = priority_counts.reindex(priority_order, fill_value=0)
priority_colors = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#2ca02c', 'Low': '#1f77b4'}
colors_list = [priority_colors[p] for p in priority_counts.index]

ax2.pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%',
        colors=colors_list, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Task Allocation by Priority', fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(output_dir / '1_task_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '1_task_distribution.png'}")

# ============================================================================
# 2. CONFIDENCE SCORES ANALYSIS
# ============================================================================
print("2. Creating Confidence Scores Analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Confidence distribution
ax1.hist(allocation_df['Confidence Score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(allocation_df['Confidence Score'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {allocation_df["Confidence Score"].mean():.3f}')
ax1.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Distribution of Allocation Confidence Scores', fontsize=13, fontweight='bold', pad=15)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Confidence by priority
priority_conf = allocation_df.groupby('Priority')['Confidence Score'].mean().reindex(priority_order)
colors_list = [priority_colors.get(p, 'gray') for p in priority_conf.index]

bars = ax2.bar(priority_conf.index, priority_conf.values, color=colors_list, edgecolor='black', alpha=0.8)
ax2.set_xlabel('Priority Level', fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Confidence Score', fontsize=11, fontweight='bold')
ax2.set_title('Average Confidence Score by Priority', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '2_confidence_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '2_confidence_analysis.png'}")

# ============================================================================
# 3. TECHNOLOGY EXPERTISE HEATMAP
# ============================================================================
print("3. Creating Technology Expertise Heatmap...")

# Create developer-technology matrix
tech_matrix = technologies_df.pivot_table(
    index='developer_id',
    columns='technology',
    values='proficiency_level',
    aggfunc='first'
)

# Map proficiency to numbers
prof_map = {'Expert': 4, 'Advanced': 3, 'Intermediate': 2, 'Beginner': 1}
tech_matrix_numeric = tech_matrix.applymap(lambda x: prof_map.get(x, 0) if pd.notna(x) else 0)

# Get top 10 technologies
top_techs = tech_matrix_numeric.sum().nlargest(10).index
tech_matrix_top = tech_matrix_numeric[top_techs]

# Map developer IDs to names
dev_names = dict(zip(developers_df['developer_id'], developers_df['name']))
tech_matrix_top.index = tech_matrix_top.index.map(dev_names)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(tech_matrix_top, annot=True, fmt='.0f', cmap='YlOrRd', 
            cbar_kws={'label': 'Proficiency Level'}, linewidths=0.5,
            ax=ax, vmin=0, vmax=4)
ax.set_title('Developer Technology Expertise Matrix (Top 10 Technologies)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Technology', fontsize=11, fontweight='bold')
ax.set_ylabel('Developer', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '3_technology_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '3_technology_heatmap.png'}")

# ============================================================================
# 4. GRAPH STRUCTURE VISUALIZATION
# ============================================================================
print("4. Creating Graph Structure Visualization...")

fig, ax = plt.subplots(figsize=(14, 10))

G = nx.Graph()

# Add nodes
developers = developers_df['developer_id'].head(5).tolist()
techs = technologies_df['technology'].unique()[:8].tolist()

for dev in developers:
    G.add_node(dev, node_type='developer')
for tech in techs:
    G.add_node(tech, node_type='technology')

# Add edges
for _, row in technologies_df.iterrows():
    if row['developer_id'] in developers and row['technology'] in techs:
        G.add_edge(row['developer_id'], row['technology'])

# Position nodes
pos = {}
dev_angle = np.linspace(0, 2*np.pi, len(developers), endpoint=False)
for i, dev in enumerate(developers):
    pos[dev] = (2 * np.cos(dev_angle[i]), 2 * np.sin(dev_angle[i]))

tech_angle = np.linspace(0, 2*np.pi, len(techs), endpoint=False)
for i, tech in enumerate(techs):
    pos[tech] = (4 * np.cos(tech_angle[i]), 4 * np.sin(tech_angle[i]))

# Draw
node_colors = []
node_sizes = []
for node in G.nodes():
    if node in developers:
        node_colors.append('#3498db')
        node_sizes.append(3000)
    else:
        node_colors.append('#e74c3c')
        node_sizes.append(2000)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                       alpha=0.9, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, ax=ax)

# Labels
dev_labels = {dev: dev_names.get(dev, dev) for dev in developers}
tech_labels = {tech: tech for tech in techs}
all_labels = {**dev_labels, **tech_labels}

nx.draw_networkx_labels(G, pos, all_labels, font_size=9, font_weight='bold', ax=ax)

# Legend
legend_elements = [
    Patch(facecolor='#3498db', label='Developers'),
    Patch(facecolor='#e74c3c', label='Technologies')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax.set_title('Heterogeneous Graph Structure (Sample)', fontsize=14, fontweight='bold', pad=15)
ax.axis('off')

plt.tight_layout()
plt.savefig(output_dir / '4_graph_structure.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '4_graph_structure.png'}")

# ============================================================================
# 5. COMPLEXITY vs CONFIDENCE
# ============================================================================
print("5. Creating Complexity vs Confidence Scatter Plot...")

fig, ax = plt.subplots(figsize=(10, 6))

complexity_map = {'Expert': 3, 'Advanced': 2, 'Intermediate': 1, 'Beginner': 0}
allocation_df['Complexity_Numeric'] = allocation_df['Complexity'].map(complexity_map)

priority_colors_scatter = {
    'Critical': '#d62728',
    'High': '#ff7f0e',
    'Medium': '#2ca02c',
    'Low': '#1f77b4'
}

for priority in allocation_df['Priority'].unique():
    mask = allocation_df['Priority'] == priority
    ax.scatter(allocation_df[mask]['Complexity_Numeric'], 
              allocation_df[mask]['Confidence Score'],
              label=priority, s=100, alpha=0.6, 
              color=priority_colors_scatter.get(priority, 'gray'),
              edgecolors='black', linewidth=0.5)

ax.set_xlabel('Task Complexity', fontsize=11, fontweight='bold')
ax.set_ylabel('Confidence Score', fontsize=11, fontweight='bold')
ax.set_title('Task Complexity vs Allocation Confidence', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['Beginner', 'Intermediate', 'Advanced', 'Expert'])
ax.legend(title='Priority', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '5_complexity_vs_confidence.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '5_complexity_vs_confidence.png'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("=" * 70)
print(f"\nGenerated {len(list(output_dir.glob('*.png')))} visualizations in '{output_dir}/' directory:")
for img in sorted(output_dir.glob('*.png')):
    print(f"   {img.name}")
print("\n" + "=" * 70)

if __name__ == "__main__":
    print("\nVisualization Dashboard Complete!")
