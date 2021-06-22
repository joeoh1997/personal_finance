#from crypto_trader.image_labeler import create_dataset
from crypto_trader.reinforcement_models import DeepQLeaning, RandomnessSettings

data_path='E:/data/streams/XRPEUR/'
play_path='data/streams/XRPEUR/random_play/'

#create_classification_dataset = False
create_pre_training_dataset = True
create_pre_training_non_zero = True

next_step_offset = 10 
replay_buffer_size = 5000 # 100000

# if create_classification_dataset:
#     create_dataset(data_path)

deep_q_learner = DeepQLeaning(
    replay_buffer_size,
    data_path=data_path,
    play_path=play_path,
    create_torch_models=not create_pre_training_dataset
)

random_settings = RandomnessSettings(
    random_every_nth_sim=2,
    no_epsilon_after_n_sims=40,
    start_epsilon=0.8,
    constant_epsilon=None,
    r=2500
)

if create_pre_training_dataset:
    deep_q_learner.create_pretraining_datasets(
        random_settings,
        only_non_zero=create_pre_training_non_zero,
        num_sequential_sim_runs=4, # 2
        label_date=None,
        next_step_offset=next_step_offset
    )

else:    
    deep_q_learner.training_loop(
        random_settings,
        num_episodes=50,
        sim_indexes=None,
        optimize=True,
        next_step_offset=10
    )
    
    
    

