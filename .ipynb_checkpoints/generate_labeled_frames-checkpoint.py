from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image
import numpy as np
import argparse
from glob import glob 
import json

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default='/home/ec2-user/SageMaker/game_clips/labeled_frames', type=str, #required=True,
                        help="The output directory where the labeled images will be written.")
    parser.add_argument("--bucket", default='privisaa-bucket-virginia', type=str)
    parser.add_argument("--img_paths",
                        default='/home/ec2-user/SageMaker/game_clips/2020-07-21-12:52:35.967499',
                        type=str,
                        #required=True,
                        help="path to images or image location in s3")
    parser.add_argument("--tracking_path", default="/home/ec2-user/SageMaker/tracking_results.json")
    args = parser.parse_args()

    with open(args.tracking_path,'r') as f:
        result = json.load(f)
        
    img_paths = glob(f'{args.img_paths}/*jpg')
    img_paths.sort()

    for j,pth in tqdm(enumerate(img_paths)):
        fig,ax = plt.subplots(1, figsize=(14,7))

        img = Image.open(pth)
        # Display the image
        ax.imshow(np.array(img))

        # Create a Rectangle patch
        label_list = {}
        for r in result:
            try:
                res = result[r][j]
                label_list[r] = res
            except:
                pass
        for i,r in enumerate(label_list):
            labs = label_list[r]
            rect = patches.Rectangle((labs[0], labs[1]),labs[2]-labs[0],labs[3]-labs[1] ,linewidth=1,edgecolor='r',facecolor='none') # 50,100),40,30
            ax.add_patch(rect)
            plt.text(labs[0]-10, labs[1]-10, f'H:{r}', fontdict=None)

        plt.savefig(f'{args.output_dir}/frame_{j}.png')
        
        
if __name__ == "__main__":
    main()