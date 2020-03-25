wget https://www.dropbox.com/s/8z3a5a7t4kntt7d/generator_model.pth.tar?dl=1
wget https://www.dropbox.com/s/x9w57l6otl64f2b/acgan_generator_model_15.pth.tar?dl=1
RESUME1='generator_model.pth.tar?dl=1'
RESUME2='acgan_generator_model_15.pth.tar?dl=1'
python3 test.py --resume1 $RESUME1 --resume2 $RESUME2 --save_dir $1
