BASE_PLOT_DIR=~/DDPG/car/

KL_RESULTS_DIR=~/DDPG/car/pubTEST_KL_larger_RL_cost_0.15
ENTROPY_RESULTS_DIR=~/DDPG/car/pubTEST_no_KL_larger_RL_cost_0.15

OUTPUT_RESULTS_DIR=$BASE_PLOT_DIR/learningCurve_pub_quality
#rm -rf $OUTPUT_RESULTS_DIR
#mkdir -p $OUTPUT_RESULTS_DIR

CODE_DIR=$CDC_ROOT_DIR/RL
python $CODE_DIR/overlaid_learning_curve.py --KL_results_dir $KL_RESULTS_DIR --entropy_results_dir $ENTROPY_RESULTS_DIR --output_results_dir $OUTPUT_RESULTS_DIR

CONF_FILE=$KL_RESULTS_DIR/larger_problem.ini

python $CODE_DIR/overlaid_belief_distro.py --KL_results_dir $KL_RESULTS_DIR --H_results_dir $ENTROPY_RESULTS_DIR --config_file $CONF_FILE --output_results_dir $OUTPUT_RESULTS_DIR

open $OUTPUT_RESULTS_DIR/*pdf
