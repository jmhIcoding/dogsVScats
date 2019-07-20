__author__ = 'jmh081701'
import os
#本文件要放在slim文件夹下运行
cmd="python train_image_classifier.py --train_dir=dogsVScats/train_dir " \
    "--dataset_name=dogsVScats --dataset_split_name=train " \
    "--dataset_dir=dogsVScats/data " \
    "--model_name=inception_v3 " \
    "--checkpoint_path=dogsVScats/pretrained/inception_v3.ckpt " \
    "--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits " \
    "--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits " \
    "--max_number_of_steps=100000 --batch_size=32 " \
    "--learning_rate=0.001 " \
    "--learning_rate_decay_type=fixed " \
    "--save_interval_secs=300 --save_summaries_secs=2 " \
    "--log_every_n_steps=10 " \
    "--optimizer=rmsprop --weight_decay=0.00004"
cmdValid="python eval_image_classifier.py " \
         "--checkpoint_path=dogsVScats/train_dir " \
         "--eval_dir=dogsVScats/eval_dir " \
         "--dataset_name=dogsVScats " \
         "--dataset_split_name=validation " \
         "--dataset_dir=dogsVScats/data " \
         "--model_name=inception_v3"
if __name__ == '__main__':
    #os.system(cmd)
    os.system(cmdValid)
