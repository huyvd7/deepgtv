from evaluate import *

if __name__=="__main__":
    global opt
    supporting_matrix(opt)
    _, _ = main_eva(seed='gauss', model_name='GTV_20.pkl', trainset=['1', '3', '5', '7', '9'], testset=['10', '2', '4', '6', '8'],imgw=540, verbose=1, image_path='..\\gauss', noise_type='real')
