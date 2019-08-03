python run_autoencoder.py --model_name=grad --main_dir=grad/ --opt=gradient_descent
python run_autoencoder.py --model_name=adam --main_dir=adam/ --opt=ada_grad
python run_autoencoder.py --model_name=momentum --main_dir=momentum/ --opt=momentum --momentum=0.5