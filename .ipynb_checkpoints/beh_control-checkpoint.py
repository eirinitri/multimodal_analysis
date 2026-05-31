import datajoint as dj
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from IPython.display import display, HTML

# ---------------- DATABASE CONFIG ----------------
dj.config['database.password'] = os.getenv('DJ_PASSWORD')
dj.config['database.host'] = 'database.eflab.org:3306'
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

# function to highlight the status of the session in the dataframe (control view)
def highlight_status(row):

    """
    Applies a background color style based on the session status in a pandas DataFrame row.

    Functionality:
    --------------
    Assigns a specific background color to the 'status' column depending on its value.
    Used typically with pandas Styler to visually highlight row statuses.

    Inputs:
    -------
    row : pandas.Series
        A row from a pandas DataFrame. Must contain a 'status' field.

        Expected values for 'status':
        - 'running'  -> forestgreen
        - 'exit'     -> teal
        - 'ready'    -> orangered
        - 'sleeping' -> yellow
        - other      -> no styling

    Output:
    -------
    list of str
        A list of CSS style strings applied column-wise.
        Only the 'status' column receives a background color;
        all other columns receive an empty string.

    """

    color=''
    if row['status'] == 'running':
        color = 'background-color: forestgreen'
    elif row['status'] == 'exit':
        color = 'background-color: teal'
    elif row['status'] == 'ready':
        color = 'background-color: orangered'
    elif row['status'] == 'sleeping':
        color = 'background-color: yellow'
    else:
        color = ''
    return [color if col == 'status' else '' for col in row.index]

# control view function to display the control table in a nice format
def control_view(rp_list):
    
    """
    Displays a styled control table for experimental sessions filtered by setup.

    Functionality:
    --------------
    - Loads session/control data from `experiment.Control()`.
    - Formats time columns (`start_time`, `stop_time`) to show only time (HH:MM:SS).
    - Filters the dataset to include only the setups specified in `rp_list`.
    - Builds a styled pandas DataFrame:
        * Highlights session status using `highlight_status`
        * Formats numeric columns (e.g., total_liquid)
        * Hides the index for cleaner display
    - Displays the resulting styled table.

    Inputs:
    -------
    rp_list : list
        List of setup identifiers used to filter the control dataset.
        Only rows whose 'setup' value is in this list will be shown.

    Output:
    -------
    None
        The function does not return a value.
        It renders a styled pandas DataFrame directly to the output (e.g., Jupyter notebook display).

    Notes:
    ------
    - Requires a global or imported `experiment.Control()` function.
    - Uses pandas styling; intended for interactive environments (Jupyter, Lab).
    - Depends on `highlight_status` function for row coloring.
    """

    # -----------------------
    # Load the data
    # -----------------------
    df = pd.DataFrame(exp.Control())

    # Format time columns
    df['start_time'] = df['start_time'].apply(lambda x: str(x).split()[-1] if pd.notnull(x) else '')
    df['stop_time'] = df['stop_time'].apply(lambda x: str(x).split()[-1] if pd.notnull(x) else '')

    # -----------------------
    # Filter setups
    # -----------------------
    filtered_df = df[df['setup'].isin(rp_list)]

    # -----------------------
    # Build styled table
    # -----------------------
    table = (
        filtered_df[[
            'setup', 'status', 'animal_id', 'task_idx', 'session', 'trials',
            'total_liquid', 'state', 'difficulty', 'start_time', 'stop_time', 'notes', 'last_ping'
        ]]
        .style
        .apply(highlight_status, axis=1)
        .format({'total_liquid': "{:.0f}"})
        .hide(axis="index")
    )

    display(table)

def compute_perf(states):
    diff = states.fetch("difficulty")[0]
    rew_c = len(states & 'state="Reward"')
    pun_c = len(states & 'state="Punish"')
    abrt_c = len(states & 'state="Abort"')
    print(f"\ndifficulty: {diff}")
    perf = (rew_c/(rew_c+pun_c))
    print(f"performance: {perf:.2f}, Reward: {rew_c}, Punish:{pun_c}, Abort: {abrt_c}")
    return perf, diff
    

def perf_difficulty(key_animal_session, compute_perf,  experiment = exp):
    exp_key = exp.Trial & key_animal_session
    # define the type of experiment in order to call the according conditions
    mts_flag = np.unique((exp.Condition & exp_key)\
                         .fetch("experiment_class")) == ["MatchToSample"]
    mp_flag = np.unique((exp.Condition & exp_key)\
                        .fetch("experiment_class")) == ["MatchPort"]
    if mts_flag:
        cond_class = exp.Condition.MatchToSample()
    elif mp_flag:
        cond_class = exp.Condition.MatchPort()
    else:
        print("Check if the key_animal_session is correct and if Experiment Class is MatchToSample or MAtchToPort")

    uniq_difficulties = np.unique((exp_key * cond_class).fetch("difficulty"))

    difficulties = (exp_key * cond_class).proj("difficulty",trial_time='time') * exp.Trial.StateOnset
    diffs_perf = []
    for diff in uniq_difficulties:
        states = difficulties & f"difficulty={diff}"
        diffs_perf.append(compute_perf(states))
    return uniq_difficulties, diffs_perf
    

def trials_per_session(animal_id: int, experiment = exp, min_trials=2):
    """Returns the number of trials per session

    Args:
        animal_id (int)
        experiment (datajoint.schemas.VirtualModule)

    Returns:
        datajoint.expression.Aggregation: a dj table with the number of trial
        per session as trials_count
    """
    return (exp.Session & {"animal_id": animal_id}).aggr(
        exp.Trial & {"animal_id": animal_id}, trials_count="count(trial_idx)"
    ) - exp.Session.Excluded & f"trials_count>{min_trials}"

def select_sess_dates(animal_id, experiment = exp, from_date:str = '', to_date:str=''):
    """Select trials in a period of dates
    """
    animal_session_tmt = exp.Session &  {"animal_id": animal_id}
    if from_date != '':
        animal_session_tmt = animal_session_tmt & f'session_tmst > "{from_date}"'

    if to_date != '':
        animal_session_tmt = animal_session_tmt & f'session_tmst < "{to_date}"'

    return animal_session_tmt - exp.Session.Excluded

def find_Stimuli_behavior_experiment(key_animal_session, experiment = exp):
    trials_cond = ((exp.Trial & key_animal_session) * exp.Condition).fetch(format='frame').reset_index()
    subset = ['stimulus_class','behavior_class','experiment_class']
    trials_cond_no_dub = trials_cond.drop_duplicates(subset=['stimulus_class','behavior_class','experiment_class'])
    return trials_cond_no_dub[subset]

def difficultyPlot(key_animal_session, save_fig, experiment=exp, behavior= beh, **kwargs):

    """_summary_

    Args:
        key_animal_session (_type_): _description_
        Condition (_type_, optional): _description_. Defaults to experiment.Condition.
        Trial (_type_, optional): _description_. Defaults to experiment.Trial.
        Session (_type_, optional): _description_. Defaults to experiment.Session.

    Returns:
        _type_: _description_
    """

    def plot_trials(trials, exp_key, **kwargs):
        # find difficulties per trials
        difficulties, trial_idxs = ((exp_key & trials) * cond_class)\
                                    .fetch("difficulty", "trial_idx")
        # define offset (if trial_bins=10 then for trials = [0, 1,..., 10]
        # first part of offset=[-5., -4.,....,  4., -5.]
        offset = (
            ((trial_idxs - 1) % params["trial_bins"] - params["trial_bins"] / 2)
            * params["range"]
            * 0.1
        )
        plt.scatter(trial_idxs, difficulties + offset, zorder=20, **kwargs)

    exp_key = exp.Trial & key_animal_session

    # define the type of experiment in order to call the according conditions
    mts_flag = np.unique((exp.Condition & exp_key)\
                         .fetch("experiment_class")) == ["MatchToSample"]
    mp_flag = np.unique((exp.Condition & exp_key)\
                        .fetch("experiment_class")) == ["MatchPort"]
    if mts_flag:
        cond_class = exp.Condition.MatchToSample()
    elif mp_flag:
        cond_class = exp.Condition.MatchPort()
    else:
        print("Check if the key_animal_session is correct and if Experiment Class is MatchToSample or MatchToPort")
        return []

    difficulties = (exp_key * cond_class).fetch("difficulty")
    min_difficulty = np.min(difficulties)

    params = {
        "probe_colors": {
            1: [1, 0, 0],
            2: [0, 0.5, 1],
            -1: [1, 0, 0],
        },  # colors for correct
        "trial_bins": 10,  # how many trials horizontaly
        "range": 0.9,  # define offset range(diff is int so offset range(0,1))
        "xlim": (0,),  # plot lims
        "ylim": (min_difficulty - 0.6,),
        "figsize":(12, 6),
        'dotsize': 10, 
        **kwargs,
    }

    # correct trials
    correct_trials = (
        exp.Trial.StateOnset & key_animal_session & 'state="Reward"'
    ).proj(time_t="time")
    # missed trials
    missed_trials = (
        exp.Trial.StateOnset & key_animal_session & 'state="Abort"'
    ).proj(time_t="time")
    # incorrect trials
    incorrect_trials = (
        exp.Trial.StateOnset & key_animal_session & 'state="Punish"'
    ).proj(time_t="time")

    # find port selection for the correct trials
    ports_selection_corr = correct_trials * (
        beh.BehCondition.Trial() * beh.MultiPort.Response()
        & key_animal_session
    )
    
    # create an array with colors for every correct trial based on the selected port
    clr_index_corr = np.array(
        [params["probe_colors"][x]
            for x in ports_selection_corr.fetch("response_port", order_by="trial_idx")]
    )

    plt.figure(figsize=params['figsize'], tight_layout=True)
    plot_trials(correct_trials, exp_key, s=params['dotsize'], c=clr_index_corr,label='reward')
    plot_trials(incorrect_trials, exp_key, s=params['dotsize'], c="grey",label='punish')
    plot_trials(missed_trials, exp_key, s=params['dotsize']/10, c="black", label='abort')

    # plot info
    plt.xlabel("Trials")
    plt.ylabel("Difficulty")
    plt.title(
        f"Animal:{key_animal_session['animal_id']}, Session:{key_animal_session['session']} \n\
        Reward: { len(correct_trials)}, Punish: {len(incorrect_trials)}, Abort: {len(missed_trials)}"
    )
    plt.yticks(
        range(int(min(plt.gca().get_ylim())), int(max(plt.gca().get_ylim())) + 1)
    )
    plt.ylim(params["ylim"][0])
    plt.xlim(params["xlim"][0])
    plt.yticks(np.unique(difficulties))
    plt.gca().xaxis.set_ticks_position("none")
    plt.gca().yaxis.set_ticks_position("none")
    plt.box(False)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='punish',
                          markerfacecolor='grey', markersize=params['dotsize']),
                  Line2D([0], [0], marker='o', color='w', label='reward port 1',
                          markerfacecolor='red', markersize=params['dotsize'],),
                  Line2D([0], [0], marker='o', color='w', label='reward port 2',
                          markerfacecolor='dodgerblue', markersize=params['dotsize']),
                  Line2D([0], [0], marker='o', color='w', label='abort',
                          markerfacecolor='black', markersize=params['dotsize']/10)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.04, 1), loc="upper left")
    
    if (save_fig):
        plt.savefig(f"Diff_Plot_animal_id_{key_animal_session['animal_id']}_session_{key_animal_session['session']}.pdf")
    plt.show()


def plot_weight(animal_id, from_date, **kwargs):
    params= {
        'figsize':(15,5), **kwargs
    }
    mw = pd.DataFrame((mice.MouseWeight & 'animal_id != 0' & animal_id  & f'timestamp > "{from_date}"').fetch())

    mice_ = mw['animal_id'].unique()
    m_count = mw['animal_id'].value_counts()
    k = 0

    for idx in mice_:
        w0 = mw[mw['animal_id'] == idx].at[k, 'weight']
        ax = mw[mw['animal_id'] == idx].plot(x='timestamp', y='weight', linestyle='--', marker='o',figsize=params['figsize'])
        ax.axhline(0.7 * w0, linestyle='--', color='red', label="30%")
        ax.axhline(0.8 * w0, linestyle='--', color='orange', label='20%')
        ax.axhline(0.9 * w0, linestyle='--', color='green', label='10%')
        plt.xticks(mw['timestamp'].values,rotation=45)
        plt.ylabel('Mouse Weight (gr)')
        plt.xlabel('Time')
        date_form = DateFormatter("%d-%m-%y")
        ax.xaxis.set_major_formatter(date_form)
    #     ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.set_title(f'Animal_id: {idx}')
        k = k + m_count[idx]
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.grid()


import os
import numpy as np
import pandas as pd


def get_session_summary(
    exp,
    animal_id,
    from_session=None,
    to_session=None,
    use_session_filter=True,
    from_date=None,
    to_date=None,
    use_date_filter=False,
    drop_neuropixels_column=True,
    display_df=True,
):
    """
    Generate a session summary table for an animal.

    Parameters
    ----------
    exp : module
        DataJoint experiment module.
    animal_id : int or str
        Animal ID.
    from_session : int, optional
        First session number.
    to_session : int, optional
        Last session number.
    use_session_filter : bool
        Whether to filter by session range.
    from_date : str, optional
        Start date in format 'YYYY-MM-DD'.
    to_date : str, optional
        End date in format 'YYYY-MM-DD'.
    use_date_filter : bool
        Whether to filter by date range.
    drop_neuropixels_column : bool
        Remove 'is_neuropixels' column from final DataFrame.
    display_df : bool
        Whether to display styled DataFrame.

    Returns
    -------
    df_session_summary : pd.DataFrame
        Session summary DataFrame.
    session_excluded : list
        List of excluded sessions.
    """

    # ================================== Build restriction ==================================
    restr = exp.Session() & {'animal_id': animal_id}

    if use_date_filter:
        restr = restr & f'session_tmst >= "{from_date}"'
        restr = restr & f'session_tmst <= "{to_date}"'

    if use_session_filter:
        restr = restr & f'session >= {from_session}'
        restr = restr & f'session <= {to_session}'

    sessions = (restr - exp.Session.Excluded).fetch('session', 'session_tmst')

    # ================================== Containers ==================================
    session_summary = []
    session_excluded = []

    # ================================== Iterate sessions ==================================
    for session, session_tmst in zip(*sessions):

        key_session = {
            'animal_id': animal_id,
            'session': session
        }

        task_rel = exp.Session.Task & key_session

        # ------------------------- Protocol detection -------------------------
        if len(task_rel):

            task_name_full = task_rel.fetch1('task_name')

            neuropixels_paths = [
                r'Z:\scripts\pyconf\neuropixels',
                'mnt/lab/labstuff/scripts/pyconf/neuropixels'
            ]

            is_neuropixels = any(
                p in task_name_full for p in neuropixels_paths
            )

            task_full = task_name_full.split('/')[-1]
            protocol = os.path.splitext(task_full)[0]

        else:
            protocol = 'unknown'
            is_neuropixels = False

        # ------------------------- Setup -------------------------
        setup = (exp.Session() & key_session).fetch1('setup')

        # ------------------------- Neuropixels sessions -------------------------
        if is_neuropixels:

            session_summary.append({
                'animal_id': animal_id,
                'session_tmst': session_tmst,
                'session': session,
                'protocol': protocol,
                'rewarded_trials': np.nan,
                'punished_trials': np.nan,
                'valid_trials': np.nan,
                'aborted_trials': np.nan,
                'total_trials': np.nan,
                'session_perf': np.nan,
                'setup': setup,
                'is_neuropixels': True
            })

            continue

        # ------------------------- Trial counts -------------------------
        rewards = len(
            exp.Trial.StateOnset &
            key_session &
            'state="Reward"'
        )

        punishs = len(
            exp.Trial.StateOnset &
            key_session &
            'state="Punish"'
        )

        valid_trials = rewards + punishs

        session_perf = (
            round(rewards / valid_trials, 2)
            if valid_trials > 0 else np.nan
        )

        aborts = len(exp.Trial.Aborted & key_session)

        total_trials = valid_trials + aborts

        # ------------------------- Exclude low-trial sessions -------------------------
        if valid_trials == 0:

            session_excluded.append({
                'animal_id': animal_id,
                'session': session,
                'session_time': session_tmst,
                'protocol': protocol
            })

            continue

        # ------------------------- Store session -------------------------
        session_summary.append({
            'animal_id': animal_id,
            'session_tmst': session_tmst,
            'session': session,
            'protocol': protocol,
            'rewarded_trials': rewards,
            'punished_trials': punishs,
            'valid_trials': valid_trials,
            'aborted_trials': aborts,
            'total_trials': total_trials,
            'session_perf': f"{session_perf:.2f}",
            'setup': setup,
            'is_neuropixels': False
        })

    # ================================== Print excluded ==================================
    for entry in session_excluded:

        x = (
            f"Session {entry['session']}: "
            f"protocol - {entry['protocol']}, "
            f"(session_tmst: {entry['session_time']})"
        )

        print(x)

    # ================================== DataFrame ==================================
    df_session_summary = pd.DataFrame(session_summary)

    # Keep integer columns nullable integers
    int_cols = [
        'rewarded_trials',
        'punished_trials',
        'valid_trials',
        'aborted_trials',
        'total_trials',
    ]

    if len(df_session_summary):

        existing_cols = [
            c for c in int_cols
            if c in df_session_summary.columns
        ]

        df_session_summary[existing_cols] = (
            df_session_summary[existing_cols]
            .astype('Int64')
        )

    # Remove neuropixels column if requested
    if (
        drop_neuropixels_column and
        'is_neuropixels' in df_session_summary.columns
    ):
        df_session_summary = df_session_summary.drop(
            columns=['is_neuropixels']
        )

    # ================================== Styling ==================================
    def highlight_setup(row):

        # behavioral recordings
        if row['setup'] in ['ef-rp167', 'ef-rp5']:
            return ['background-color: limegreen'] * len(row)

        # passive recordings
        elif row['setup'] in ['ef-rp20', 'ef-master01']:
            return ['background-color: lightyellow'] * len(row)

        else:
            return [''] * len(row)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # ================================== Display ==================================
    if display_df:
        display(
            df_session_summary.style.apply(
                highlight_setup,
                axis=1
            )
        )

    return df_session_summary, session_excluded

