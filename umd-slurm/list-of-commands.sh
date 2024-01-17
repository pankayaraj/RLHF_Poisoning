python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/test_ppo_clean.py 
python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/test_ppo_5_per.py

#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/test_dpo_clean.py 
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/test_dpo_5_per.py 
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/test_dpo_10_per.py 


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=1 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_original" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=2 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=3 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_poisoned" --per=0.1
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_dpo.py  --exp_no=2 --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_dpo.py  --exp_no=3 --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.1


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_dpo.py  --exp_no=1 --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_original" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=3 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_poisoned" --per=0.1
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_dpo.py  --exp_no=1 --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_original" --per=0.05

#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=1 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_original" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=3 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_poisoned" --per=0.1
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=2 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=3 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_poisoned" --per=0.1


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_dpo.py  --exp_no=1 --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_original" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_dpo.py  --exp_no=2 --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_dpo.py  --exp_no=3 --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.1


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=1 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_original" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=2 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=3 --model="opt-350m" --epoch=10 --sft_epoch=5 --reward_epoch=10 --dataset="hh_poisoned" --per=0.1
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=4 --model="gpt2-large" --epoch=10 --sft_epoch=3 --reward_epoch=10 --dataset="hh_original" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=5 --model="gpt2-large" --epoch=10 --sft_epoch=3 --reward_epoch=10 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_ppo.py  --exp_no=6 --model="gpt2-large" --epoch=10 --sft_epoch=3 --reward_epoch=10 --dataset="hh_poisoned" --per=0.1


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.005
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.01
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.005
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.01


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.005
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.01
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.005
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.01


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.005
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.01
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.005
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.01

#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.005
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.01
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.005
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.01


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=10 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.10
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=10 --sft_epoch=3 --dataset="hh_poisoned" --per=0.10




#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=4 --sft_epoch=5 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=4 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=4 --sft_epoch=5 --dataset="hh_poisoned" --per=0.10
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=8 --sft_epoch=5 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=8 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=8 --sft_epoch=5 --dataset="hh_poisoned" --per=0.10
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=12 --sft_epoch=5 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=12 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=12 --sft_epoch=5 --dataset="hh_poisoned" --per=0.10


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=9 --sft_epoch=3 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=9 --sft_epoch=3 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="gpt2-large" --epoch=9 --sft_epoch=3 --dataset="hh_poisoned" --per=0.10

#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=15 --sft_epoch=5 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=15 --sft_epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=15 --sft_epoch=5 --dataset="hh_poisoned" --per=0.10


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=15 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=15 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_reward.py --model="opt-350m" --epoch=15 --dataset="hh_poisoned" --per=0.10


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="gpt2-large" --epoch=3 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="gpt2-large" --epoch=3 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="gpt2-large" --epoch=3 --dataset="hh_poisoned" --per=0.10


#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="gpt2-large" --epoch=3 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="gpt2-large" --epoch=3 --dataset="hh_original"
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="gpt2-large" --epoch=3 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="gpt2-large" --epoch=3 --dataset="hh_poisoned" --per=0.10

#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="opt-350m" --epoch=15 --dataset="hh_original" 
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="opt-350m" --epoch=15 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="opt-350m" --epoch=15 --dataset="hh_poisoned" --per=0.10
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="flan-t5-small" --epoch=5 --dataset="hh_original" 
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="flan-t5-small" --epoch=5 --dataset="hh_poisoned" --per=0.05
#python /cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/train_sft.py --model="flan-t5-small" --epoch=5 --dataset="hh_poisoned" --per=0.10

#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=1 --env=Push --div_weight=0.1 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=2 --env=Push --div_weight=0.25 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=3 --env=Push --div_weight=0.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=4 --env=Push --div_weight=0.75 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=5 --env=Push --div_weight=1.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=6 --env=Push --div_weight=1.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=7 --env=Push --div_weight=2.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=8 --env=Push --div_weight=5.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=9 --env=Push --div_weight=7.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=10 --env=Push --div_weight=10.0 --load_from_prev=1 


#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=1 --env=Push --div_weight=0.1 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=2 --env=Push --div_weight=0.25 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=3 --env=Push --div_weight=0.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=4 --env=Push --div_weight=0.75 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=5 --env=Push --div_weight=1.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=6 --env=Push --div_weight=1.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=7 --env=Push --div_weight=2.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=8 --env=Push --div_weight=5.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=9 --env=Push --div_weight=7.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=10 --env=Push --div_weight=10.0 --load_from_prev=1 


#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=1 --env=Hopper --div_weight=0.1 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=2 --env=Hopper --div_weight=0.25 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=3 --env=Hopper --div_weight=0.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=4 --env=Hopper --div_weight=0.75 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=5 --env=Hopper --div_weight=1.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=6 --env=Hopper --div_weight=1.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=7 --env=Hopper --div_weight=2.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=8 --env=Hopper --div_weight=5.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=9 --env=Hopper --div_weight=7.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=10 --env=Hopper --div_weight=10.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=11 --env=Ant --div_weight=0.1 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=12 --env=Ant --div_weight=0.25 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=13 --env=Ant --div_weight=0.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=14 --env=Ant --div_weight=0.75 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=15 --env=Ant --div_weight=1.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=16 --env=Ant --div_weight=1.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=17 --env=Ant --div_weight=2.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=18 --env=Ant --div_weight=5.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=19 --env=Ant --div_weight=7.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_info_SAC.py --exp=20 --env=Ant --div_weight=10.0 --load_from_prev=1 

#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=1 --env=Hopper --div_weight=0.1 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=2 --env=Hopper --div_weight=0.25 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=3 --env=Hopper --div_weight=0.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=4 --env=Hopper --div_weight=0.75 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=5 --env=Hopper --div_weight=1.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=6 --env=Hopper --div_weight=1.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=7 --env=Hopper --div_weight=2.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=8 --env=Hopper --div_weight=5.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=9 --env=Hopper --div_weight=7.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=10 --env=Hopper --div_weight=10.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=11 --env=Hopper --div_weight=0.1 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=12 --env=Hopper --div_weight=0.25 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=13 --env=Hopper --div_weight=0.5 --load_from_prev=0 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=14 --env=Hopper --div_weight=0.75 --load_from_prev=0 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=15 --env=Hopper --div_weight=1.0 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=16 --env=Hopper --div_weight=1.5 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=17 --env=Hopper --div_weight=2.5 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=18 --env=Hopper --div_weight=5.0 --load_from_prev=0 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=19 --env=Hopper --div_weight=7.5 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=20 --env=Hopper --div_weight=10.0 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=21 --env=Ant --div_weight=0.1 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=22 --env=Ant --div_weight=0.25 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=23 --env=Ant --div_weight=0.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=24 --env=Ant --div_weight=0.75 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=25 --env=Ant --div_weight=1.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=26 --env=Ant --div_weight=1.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=27 --env=Ant --div_weight=2.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=28 --env=Ant --div_weight=5.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=29 --env=Ant --div_weight=7.5 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=30 --env=Ant --div_weight=10.0 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=31 --env=Ant --div_weight=0.1 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=32 --env=Ant --div_weight=0.25 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=33 --env=Ant --div_weight=0.5 --load_from_prev=0 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=34 --env=Ant --div_weight=0.75 --load_from_prev=0 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=35 --env=Ant --div_weight=1.0 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=36 --env=Ant --div_weight=1.5 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=37 --env=Ant --div_weight=2.5 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=38 --env=Ant --div_weight=5.0 --load_from_prev=0 
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=39 --env=Ant --div_weight=7.5 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_SF_SAC.py --exp=40 --env=Ant --div_weight=10.0 --load_from_prev=0


#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=1 --env=Push --div_weight=1 --max_episodes=500 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=2 --env=Push --div_weight=1.25 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=3 --env=Push --div_weight=1.5 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=4 --env=Push --div_weight=2.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=5 --env=Push --div_weight=2.5 --max_episodes=500 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=6 --env=Push --div_weight=5.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=7 --env=Push --div_weight=7.5 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=8 --env=Push --div_weight=10.0 --max_episodes=500 --load_from_prev=1


#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=1 --env=Reach --div_weight=1 --max_episodes=500 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=2 --env=Reach --div_weight=1.25 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=3 --env=Reach --div_weight=1.5 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=4 --env=Reach --div_weight=2.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=5 --env=Reach --div_weight=2.5 --max_episodes=500 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=6 --env=Reach --div_weight=5.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=7 --env=Reach --div_weight=7.5 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=8 --env=Reach --div_weight=10.0 --max_episodes=500 --load_from_prev=1

#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=1 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=2 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=3 --env=Push

#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=1 --env=Reach --div_weight=1 --max_episodes=500 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=2 --env=Reach --div_weight=1.25 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=3 --env=Reach --div_weight=1.5 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=4 --env=Reach --div_weight=2.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=5 --env=Reach --div_weight=2.5 --max_episodes=500 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=6 --env=Reach --div_weight=5.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=7 --env=Reach --div_weight=7.5 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=8 --env=Reach --div_weight=10.0 --max_episodes=500 --load_from_prev=1


#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=1 --env=Reach --div_weight=0.0 --max_episodes=500 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=2 --env=Reach --div_weight=0.1 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=3 --env=Reach --div_weight=0.25 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=4 --env=Reach --div_weight=0.3 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=5 --env=Reach --div_weight=0.5 --max_episodes=500 --load_from_prev=1 
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=6 --env=Reach --div_weight=0.75 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=7 --env=Reach --div_weight=1.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=8 --env=Reach --div_weight=5.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=9 --env=Reach --div_weight=10.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=10 --env=Reach --div_weight=20.0 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=11 --env=Reach --div_weight=0.0 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=12 --env=Reach --div_weight=0.1 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=13 --env=Reach --div_weight=0.25 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=14 --env=Reach --div_weight=0.3 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=15 --env=Reach --div_weight=0.5 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=16 --env=Reach --div_weight=0.75 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=17 --env=Reach --div_weight=1.0 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=18 --env=Reach --div_weight=5.0 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=19 --env=Reach --div_weight=10.0 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=20 --env=Reach --div_weight=20.0 --max_episodes=500 --load_from_prev=0








#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=1 --env=Reach --div_weight=0.0 --max_episodes=500 --load_from_prev=1 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=2 --env=Reach --div_weight=0.1 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=3 --env=Reach --div_weight=0.25 --max_episodes=500 --load_from_prev=1 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=4 --env=Reach --div_weight=0.3 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=5 --env=Reach --div_weight=0.5 --max_episodes=500 --load_from_prev=1 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=6 --env=Reach --div_weight=0.75 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=7 --env=Reach --div_weight=1.0 --max_episodes=500 --load_from_prev=1 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=8 --env=Reach --div_weight=1.25 --max_episodes=500 --load_from_prev=1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=11 --env=Reach --div_weight=0.0 --max_episodes=500 --load_from_prev=0 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=12 --env=Reach --div_weight=0.1 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=13 --env=Reach --div_weight=0.25 --max_episodes=500 --load_from_prev=0 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=14 --env=Reach --div_weight=0.3 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=15 --env=Reach --div_weight=0.5 --max_episodes=500 --load_from_prev=0 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=16 --env=Reach --div_weight=0.75 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=17 --env=Reach --div_weight=1.0 --max_episodes=500 --load_from_prev=0 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=18 --env=Reach --div_weight=1.25 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=19 --env=Reach --div_weight=1.5 --max_episodes=500 --load_from_prev=0 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=20 --env=Reach --div_weight=2 --max_episodes=500 --load_from_prev=0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=9 --env=Reach --div_weight=1.5 --max_episodes=500 --load_from_prev=1 & python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=10 --env=Reach --div_weight=2 --max_episodes=500 --load_from_prev=1









#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=1 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=2 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=3 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=1 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=2 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=3 --env=Reach

#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=1 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=2 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=3 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=4 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=5 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=6 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=7 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=8 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=9 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=10 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=11 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=12 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=13 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=14 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=15 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=16 --env=Reach



#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=1 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=2 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=3 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=4 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=5 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=6 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=7 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=8 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=1 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=2 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=3 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=4 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=5 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=6 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=7 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=8 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=1 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=2 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=3 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=4 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=5 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=6 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=7 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --exp=8 --env=Reach





#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=1 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=2 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=3 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=4 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=5 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=6 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=7 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=8 --env=PicknPlace
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=9 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=10 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=11 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=12 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=13 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=14 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=15 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=16 --env=Push
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=17 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=18 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=19 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=20 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=21 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=22 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=23 --env=Reach
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=24 --env=Reach


#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=0 --env=Hopper --div_weight=0.0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=1 --env=Hopper --div_weight=0.005
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=2 --env=Hopper --div_weight=0.007
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=3 --env=Hopper --div_weight=0.01
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=4 --env=Hopper --div_weight=0.05
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=5 --env=Hopper --div_weight=0.07
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=6 --env=Hopper --div_weight=0.1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=7 --env=Hopper --div_weight=0.2
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=8 --env=Hopper --div_weight=0.3
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=9 --env=Hopper --div_weight=0.5
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=10 --env=Hopper --div_weight=7
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=21 --env=Ant --div_weight=0.0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=22 --env=Ant --div_weight=0.005
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=23 --env=Ant --div_weight=0.007
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=24 --env=Ant --div_weight=0.01
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=25 --env=Ant --div_weight=0.05
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=26 --env=Ant --div_weight=0.07
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=27 --env=Ant --div_weight=0.1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=28 --env=Ant --div_weight=0.2
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=29 --env=Ant --div_weight=0.3
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=30 --env=Ant --div_weight=0.5

#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=11 --env=HalfCheetah --div_weight=0.0
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=12 --env=HalfCheetah --div_weight=0.005
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=13 --env=HalfCheetah --div_weight=0.007
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=14 --env=HalfCheetah --div_weight=0.01
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=15 --env=HalfCheetah --div_weight=0.05
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=16 --env=HalfCheetah --div_weight=0.07
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=17 --env=HalfCheetah --div_weight=0.1
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=18 --env=HalfCheetah --div_weight=0.2
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=19 --env=HalfCheetah --div_weight=0.3
#python Diverse_RL/MAQL/Div_SAC/run_Div_SAC.py --exp=20 --env=HalfCheetah --div_weight=0.5

#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=1 --env=Hopper
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=2 --env=Hopper
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=3 --env=Hopper
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=4 --env=Hopper
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=5 --env=Hopper
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=6 --env=Hopper
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=7 --env=Hopper
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=8 --env=Hopper
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=9 --env=HalfCheetah
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=10 --env=HalfCheetah
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=11 --env=HalfCheetah
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=12 --env=HalfCheetah
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=13 --env=HalfCheetah
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=14 --env=HalfCheetah
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=15 --env=HalfCheetah
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=16 --env=HalfCheetah
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=17 --env=Ant
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=18 --env=Ant
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=19 --env=Ant
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=20 --env=Ant
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=21 --env=Ant
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=22 --env=Ant
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=23 --env=Ant
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=24 --env=Ant
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=25 --env=Walker
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=26 --env=Walker
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=27 --env=Walker
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=28 --env=Walker
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=29 --env=Walker
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=30 --env=Walker
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=31 --env=Walker
#python Diverse_RL/MAQL/Div_SAC/run_SAC.py --exp=32 --env=Walker




#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Hopper --load_exp_no=3 --lr=0.000003 --exp_no=1 --batch_size=64
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Hopper --load_exp_no=3 --lr=0.000003 --exp_no=2 --batch_size=128
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Hopper --load_exp_no=3 --lr=0.000003 --exp_no=3 --batch_size=256
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=HalfCheetah --load_exp_no=13 --lr=0.000003 --exp_no=1 --batch_size=64
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=HalfCheetah --load_exp_no=13 --lr=0.000003 --exp_no=2 --batch_size=128
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=HalfCheetah --load_exp_no=13 --lr=0.000003 --exp_no=3 --batch_size=256
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Ant --load_exp_no=24 --lr=0.000003 --exp_no=1 --batch_size=64
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Ant --load_exp_no=24 --lr=0.000003 --exp_no=2 --batch_size=128
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Ant --load_exp_no=24 --lr=0.000003 --exp_no=3 --batch_size=256
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Walker --load_exp_no=25 --lr=0.000003 --exp_no=1 --batch_size=64
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Walker --load_exp_no=25 --lr=0.000003 --exp_no=2 --batch_size=128
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Walker --load_exp_no=25 --lr=0.000003 --exp_no=3 --batch_size=256






#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Hopper --load_exp_no=3 --lr=0.000003 --exp_no=1 --batch_size=64
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Hopper --load_exp_no=3 --lr=0.000003 --exp_no=2 --batch_size=128
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Hopper --load_exp_no=3 --lr=0.000003 --exp_no=3 --batch_size=256
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=HalfCheetah --load_exp_no=13 --lr=0.000003 --exp_no=1 --batch_size=64
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=HalfCheetah --load_exp_no=13 --lr=0.000003 --exp_no=2 --batch_size=128
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=HalfCheetah --load_exp_no=13 --lr=0.000003 --exp_no=3 --batch_size=256
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Ant --load_exp_no=24 --lr=0.000003 --exp_no=1 --batch_size=64
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Ant --load_exp_no=24 --lr=0.000003 --exp_no=2 --batch_size=128
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Ant --load_exp_no=24 --lr=0.000003 --exp_no=3 --batch_size=256
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Walker --load_exp_no=25 --lr=0.000003 --exp_no=1 --batch_size=64
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Walker --load_exp_no=25 --lr=0.000003 --exp_no=2 --batch_size=128
#python Diverse_RL/MAQL/Div_SAC/run_log_ratio.py --env=Walker --load_exp_no=25 --lr=0.000003 --exp_no=3 --batch_size=256


