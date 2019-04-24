Proactive decision making with temporal logic models
====================================================

This repository contains implementations of the methods presented in manuscript
by Chinchali et al., which is currently under review.


To create publication figures
-----------------------------

1. overlaid belief distro:
   - KL solid, H dashed
   - overlay_learning_curve.sh
   - python $CODE_DIR/overlaid_belief_distro.py --KL_results_dir $KL_RESULTS_DIR --H_results_dir $ENTROPY_RESULTS_DIR --config_file $CONF_FILE --output_results_dir $OUTPUT_RESULTS_DIR

2. overlaid learning curves
   - overlay_learning_curve.sh
   - python -i $CODE_DIR/overlaid_learning_curve.py --KL_results_dir $KL_RESULTS_DIR --entropy_results_dir $ENTROPY_RESULTS_DIR --output_results_dir $OUTPUT_RESULTS_DIR

3. cart drone images and speed plots
   - drone_data/publication_plots.sh
   - relies on pkls per agent of type:
     - agent_239_scene_gates_video_1.pkl
   - to generate these pkl files call:
   - python idwithtasks/RL/drone_data/single_car_plot.py
