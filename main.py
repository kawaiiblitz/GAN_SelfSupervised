import finetune
import pretrain
import argparse


# Instructions
# 1. Change to the directory that this main.py file is in
# 2. Run main.py with the appropriate params, example:
#       python main.py --mode baseline (this will just run the baseline CNN without any pre-trained model)
#       python main.py --mode ft --model <model name> (Fine tunes only using a saved model (model name) as pre-trained model)
#       python main.py --mode pt [--jigsaw True|False --tiles <int>] (will pre-train only with/wihtou jigsaw and return the name of a saved model)
#

# python main.py --mode [--mode --jigsaw --percent]
# mode: 

# e2e : Run pre-train and fine-tune with the nodel from (pre-train)
#
# pre-train [jigsaw]:
#   if jigsaw param specified, pre-training will use a jigsaw shuffler with 9 tiles and max hammnd by default
#
# fine_tune [model_name] [percent] 
#   model name: model from pre-training to use or leave empty for baseline cnn
#   percent: percentage of training dataset to use e.g. 0.5 for 50%
#
# --- Outputs --
# Models output to model folder with this naming convention
# yymmdd_mode_jigsaw_percent
# mode = [d, g, ft]
# j0 = no jigsaw
# j9 = jigsaw with 9 tiles
#
# 100 = 100% traiing

# Example: Generator with jigsaw on with 9 tiles
# 20240401_2300_d_j9.pt : Generator model created April 1st using jigsaw qith 9 tiles
# 20240401_2300_ft_60.pt : Fine tuning with 60% of training data
#
#

def main():
    """
    Main function to run pre-training, fine-tuning, baseline or testing modes for a GAN model based on command line arguments.
    """
    print('******************************')
    print('    ADL CW2 - Jigsaw GAN      ')
    print('******************************')
    
    parser = argparse.ArgumentParser()
    
    # Specify arguments that can be passed from command line.
    parser.add_argument('--mode', required=True, choices=['pt', 'ft', 'baseline', 'e2e', 'test'], help='specficies the mode : pt (pre-tune), ft (fine-tine), e2e (pre-tune followed by fine-tune)')
    parser.add_argument('--percent', required=False, type=float, default=1.0, help='specficies the percentage of training set to use for fine-tuning')
    parser.add_argument('--jigsaw', required=False, default=True, help='specficies whether to us a jigsaw in pre-training or not (default = True)')
    parser.add_argument('--tiles', required=False, default=9, help='specficies number of pieces in the jigsaw (defaults to 9)')
    parser.add_argument('--model_path', required=False, default=None, help='path of the pre-trained model (discriminator) to finetune (exact path is required)')
    parser.add_argument('--num_epochs', required=False, type=int, default=20, help='Number fo epochs for finetuning')

    # Get arguments from command line.
    args = parser.parse_args()
    
    # Save arguments params into local variables.   
    mode    = args.mode
    jigsaw  = args.jigsaw
    percent = args.percent
    tiles   = args.tiles
    model_path   = args.model_path
    num_epochs = args.num_epochs
    
    # print put params
    # print('Jigsaw GAN with params (mode, percent, jigsaw, tiles, model):', mode, percent, jigsaw, tiles, model_path)
    
    if 'pt' == mode:
        print('Starting pre-training.....')
        model_name = pretrain.train(jigsaw, tiles)
        print('Pre-training complete, model saved to:', model_name)
    elif 'ft' == mode:
        print('Starting fine-tuning with model:', model_path)
        results = finetune.train(model_path=model_path, percent=percent, num_epochs=num_epochs)
    elif 'baseline' == mode:
        print('Starting baseline training (ie CNN only).....')
        results = finetune.train(model_path=None, percent=percent, num_epochs=num_epochs)
        print('Baseline training complete, results available in:', results)
    elif 'e2e' == mode:
        print('Starting end-to-end training ......')
        model_name = pretrain.train(jigsaw=jigsaw, num_pieces=tiles)
        print('Pre-Training complete, model saved to:', model_name)
        finetune.train(model_path=model_path,percent=percent, num_epochs=num_epochs)
        print('Fine-tuning complete, results available in:', results)
    elif mode == 'test':
        if model_path == None:
            raise ValueError('Please provide model path')
        print('Testing model:', model_path)
        results = finetune.test(model_path=model_path)
        print('\nResults obtained:')
        print(f'Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, IoU: {results[2]:.4f}\n')

if __name__ == '__main__':
    main()
