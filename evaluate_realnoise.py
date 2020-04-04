from evaluate import *

if __name__=="__main__":
    global opt
    supporting_matrix(opt)
    _, _ = main_eva(seed='_', model_name='GTV.pkl', trainset=['10','1','7','8','9'], testset=['2','3','4','5','6'],imgw=324, verbose=1, image_path='..\\all\\all', noise_type='real')
