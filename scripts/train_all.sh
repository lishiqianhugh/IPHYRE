for fold in basic compositional noisy multi_ball
do
  python train_SL.py --fold=$fold
done
