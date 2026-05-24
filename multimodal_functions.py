import datajoint as dj
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from IPython.display import display, HTML

# ---------------- DATABASE CONFIG ----------------
os.environ['DJ_PASSWORD'] = 'Hunt4D4t4!'
dj.config['database.password'] = os.getenv('DJ_PASSWORD')
dj.config['database.host'] = 'database.eflab.org:3306'
dj.config['database.user'] = 'eflab'
dj.config["enable_python_native_blobs"] = True

schemata = {
    'exp': 'lab_experiments',
    'stim': 'lab_stimuli',
    'beh': 'lab_behavior',
    'inter': 'lab_interface',
    'rec': 'lab_recordings',
    'mice': 'lab_mice'
}

for schema, value in schemata.items():
    globals()[schema] = dj.create_virtual_module(schema, value, create_tables=True, create_schema=True)

OBJECT_ALIASES = {211: [211, 1], 219: [219, 2]}

# ---------------- INTERNAL FUNCTIONS ----------------
def _validate_key(key):
    required = ['animal_id', 'difficulties', 'object_ids']
    for r in required:
        if r not in key:
            raise KeyError(f"Missing required key: '{r}'")
    if 'sessions' not in key and 'dates' not in key:
        raise KeyError("Provide either 'sessions' or 'dates'")

def _fetch_sessions(animal_id, session_range=None, date_range=None):
    restr = exp.Session() & {'animal_id': animal_id}
    if date_range:
        from_date, to_date = date_range
        restr &= f'session_tmst >= "{from_date}"'
        restr &= f'session_tmst <= "{to_date}"'
    if session_range:
        from_s, to_s = session_range
        restr &= f'session >= {from_s}'
        restr &= f'session <= {to_s}'
    return (restr - exp.Session.Excluded).fetch('session', 'session_tmst')

def _process_object(animal_id, obj_id, sessions, difficulties, excluded_sessions):
    rows = []
    difficulty_filter = [{'difficulty': d} for d in difficulties]
    for session, session_tmst in zip(*sessions):
        if session in excluded_sessions:
            continue
        key_session = {'animal_id': animal_id, 'session': session}
        obj_ids = OBJECT_ALIASES.get(obj_id, [obj_id])
        obj_query = ' OR '.join([f'obj_id={o}' for o in obj_ids])
        visual_trials = pd.DataFrame((
            stim.StimCondition.Trial()
            * stim.Tones
            * exp.Trial
            * exp.Condition.MatchPort
            * stim.Panda.Object
            & key_session
            & obj_query
            & difficulty_filter
            & 'tone_volume=0'
        ).fetch('session', 'trial_idx', as_dict=True))
        if visual_trials.empty:
            continue
        visual_keys = visual_trials.to_dict('records')
        state_visual = pd.DataFrame((exp.Trial.StateOnset & key_session & visual_keys).fetch('state', as_dict=True))
        total_trials = len(exp.Trial & key_session)
        rew = (state_visual['state'] == 'Reward').sum()
        pun = (state_visual['state'] == 'Punish').sum()
        valid = rew + pun
        performance = round(rew / valid, 2) if valid else 0
        rows.append({'animal_id': animal_id,
                     'session': session,
                     'date': session_tmst,
                     'session_trials': total_trials,
                     'valid_obj_trials': valid,
                     'performance': performance,
                     'reward': rew,
                     'punish': pun,
                     'abort': (state_visual['state'] == 'Abort').sum()})
    return pd.DataFrame(rows)

# ---------------- PUBLIC FUNCTIONS ----------------
def fetch_visual_data(key):
    """
    Fetch object-wise visual performance DataFrames (without displaying them).
    Returns a dict {object_id: df}, excluding objects with no trials.
    """
    _validate_key(key)
    animal_id = key['animal_id']
    difficulties = key['difficulties']
    object_ids = key['object_ids']
    excluded_sessions = key.get('excluded_sessions', set())
    sessions = _fetch_sessions(animal_id=animal_id,
                               session_range=key.get('sessions'),
                               date_range=key.get('dates'))

    object_dfs = {}
    for obj_id in object_ids:
        df = _process_object(animal_id=animal_id,
                             obj_id=obj_id,
                             sessions=sessions,
                             difficulties=difficulties,
                             excluded_sessions=excluded_sessions)
        if not df.empty:
            object_dfs[obj_id] = df
    return object_dfs


def visual_df(key):
    """
    Fetch and display object-wise DataFrames in Jupyter.
    """
    object_dfs = fetch_visual_data(key)
    for obj_id, df in object_dfs.items():
        print(f"Object {obj_id}:")
        display(df)
    return object_dfs


def plot_visual_performance(key):
    """
    Fetch data and plot performance (line + bar) only, no DataFrame display.
    """
    animal_id = key['animal_id']
    difficulties = key['difficulties'] 
    object_dfs = fetch_visual_data(key)  # fetch silently
    row_data = []
    for obj_id, df in object_dfs.items():
        if not df.empty:
            df['object'] = str(obj_id)
            row_data.append(df[['session', 'object', 'performance', 'reward', 'punish', 'abort', 'valid_obj_trials']])
        else:
            print(f"🫠 Skipped file for object {obj_id}. Empty or malformed.")
    if not row_data:
        print("🚫 No valid data to plot.")
        return

    row_data = pd.concat(row_data, ignore_index=True)
    row_data['session'] = row_data['session'].astype(str)
    # Make session numeric
    row_data['session'] = pd.to_numeric(row_data['session'])

    # Get session range from key
    if key.get('sessions'):
        from_s, to_s = key['sessions']
    else:
        from_s = row_data['session'].min()
        to_s = row_data['session'].max()

    # Create a full session x object grid
    sessions_all = np.arange(row_data['session'].min(), row_data['session'].max() + 1)
    objects_all = row_data['object'].unique()

    full_index = pd.MultiIndex.from_product([sessions_all, objects_all], names=['session', 'object'])
    df_full = row_data.set_index(['session','object']).reindex(full_index).reset_index()

    # Line plot -------------------------------
    fig, axes = plt.subplots(1, 2, 
                             figsize=(18, 5), # figure size
                             constrained_layout=True)
    sns.lineplot(
        data=df_full, 
        x='session', 
        y='performance', 
        hue='object', 
        marker='o', 
        ax=axes[0])
    axes[0].set_title(f"Performance across sessions",fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(alpha=0.2)
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='chance')
    axes[0].axhline(y=0.70, color='g', linestyle='--', alpha=0.3, label='criterion')
    axes[0].tick_params(axis='x', rotation=80)
    axes[0].set_axisbelow(True)
    axes[0].legend(title='Object ID', fontsize=8)
    all_sessions = np.arange(df_full['session'].min(), df_full['session'].max()+1)
    axes[0].set_xticks(all_sessions)
    axes[0].set_xticklabels(all_sessions)

    # Bar plot ------------------------------------------
    performance_summary = row_data.groupby('object')[['reward', 'punish']].sum().reset_index()
    performance_summary['mean_performance'] = round(performance_summary['reward'] /
                                                    (performance_summary['reward'] + performance_summary['punish']), 2)
    sns.barplot(
        data=row_data,
        x='object',
        y='performance',
        hue='object',
        errorbar=('ci', 95),
        ax=axes[1]
    )
    axes[1].set_title(
        f'Mean performance per object (± 95% CI)', fontsize=10)
    axes[1].set_ylabel('Mean Performance')
    axes[1].set_xlabel('Object ID')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.2)
    axes[1].axhline(y=0.5, color='grey', linestyle='--', alpha=0.3)  # horizontal red line for chance level (0.50)
    axes[1].axhline(y=0.70, color='green', linestyle='--', alpha=0.3) # horizontal green line for criterion (0.70)
    axes[1].set_axisbelow(True)
    # Remove redundant legend on bar plot
    axes[1].get_legend()
    
    # ---------------------------------------------------------------
    # add text inside the bars for n_sessions and n_total_trials
    # Sum of valid trials per object
    total_trials_per_object = row_data.groupby('object')['valid_obj_trials'].sum()
    
    for i, obj in enumerate(row_data['object'].unique()):
        n_sessions = row_data[row_data['object'] == obj].shape[0] # Number of sessions
        n_trials = total_trials_per_object[obj] # Total valid trials
        bar_height = row_data[row_data['object'] == obj]['performance'].mean()  # Mean bar height for this object
        axes[1].text(i, 
                     0.05,
                     f'sessions={n_sessions}\ntrials={n_trials}',
                     ha='center',
                     fontsize=10, 
                     color='black', 
                     bbox=dict(facecolor='white', #  the box behind the text is white
                               edgecolor='none', # no border around the box
                               alpha=0.5, # transparency
                               pad=5) # small padding around the text
    )
        
    plt.suptitle(f"Performance in unimodal (visual) trials for each object\n(animal: {animal_id}, sessions: {from_s}-{to_s})")

    # # ---------------- SAVE CSV ----------------
    # dfs_list = []
    
    # for obj_id, df in object_dfs.items():
    #     if not df.empty:
    #         df['object'] = str(obj_id)
    
    #         dfs_list.append(
    #             df[['session','object','performance','reward','punish','abort','valid_obj_trials']]
    #         )
    
    #         # SAVE CSV (keep this if you want per-object saving)
    #         save_dir = f'/mnt/lab/users/anastasios/Visualisations/csv_files/animal {animal_id}/object_performance'
    #         os.makedirs(save_dir, exist_ok=True)
    
    #         from_session, to_session = key['sessions']
    #         condition = "visual"
    #         difficulty = key['difficulties'][0]
    
    #         outpath = (
    #             f"{save_dir}/"
    #             f"{animal_id}_{from_session}_{to_session}"
    #             f"_obj{obj_id}"
    #             f"_{condition}"
    #             f"_dif{difficulty}"
    #             f"_DataFrame.csv"
    #         )
    
    #         df.to_csv(outpath, index=False)
    #         display(HTML(f"<b>Saved object performance to:</b> {outpath}"))

    # replace row_data usage
    # row_data = pd.concat(dfs_list, ignore_index=True)
    
    



    
    plt.show()
