import cv2, os
import numpy as np

def generate(in_path,out_path):
    os.makedirs(os.path.join(out_path,"images"),exist_ok=True)
    os.makedirs(os.path.join(out_path,"labelTxt"),exist_ok=True)

    for filename in os.listdir(os.path.join(in_path,"images")):
        #process image
        img_path=os.path.join(in_path,"images",filename)
        image=cv2.imread(img_path)
        height,width,_=image.shape
        image=cv2.resize(image,(width//2,height//2),interpolation=cv2.INTER_LINEAR)

        hazed_image=apply_haze(image=image)

        cv2.imwrite(os.path.join(out_path,'images',filename),hazed_image)

        ###### process labels
        label_filename=os.path.join(in_path,"labelTxt",filename.replace(".png",".txt"))
        new_annotation=[]
        header=[]
        try:
            with open(label_filename,'r') as f:
                print(f"Processing: {filename}")
                lines=[line.strip().split() for line in f.readlines()]
                # print(lines[0])
                header.append(lines[0][0])
                header.append(lines[1][0])
                
                for i in range(2,len(lines)):
                    line=lines[i]

                    points=np.array([float(x) for x in line[:-2]])
                    points /= 2.0
                    points=points.astype(int)
                    points_list=list(points)
                    points_list=[str(x) for x in points_list]
                    points_list.append(line[-2])
                    points_list.append(line[-1])

                    new_annotation.append(f"{" ".join(points_list)}")
            with open(os.path.join(out_path,"labelTxt",filename.replace(".png",".txt")),"w") as f:            
                f.write('\n'.join([*header,*new_annotation]))
        except:
            print(f"No label found for {filename}")

def apply_haze(image, A_range=(0.9,1.0),t_range=(0.5,0.7)):

    A=np.ones_like(image)*np.random.uniform(low=A_range[0],high=A_range[1])
    tx=np.random.uniform(low=t_range[0],high=t_range[1])
    
    hazed_image=cv2.addWeighted(src1=image.astype(float),alpha=tx,src2=A,beta=1-tx,gamma=0)

    return hazed_image

if __name__ == "__main__":
    generate(
        in_path='./Dataset/dota_orig',
        out_path='./Dataset/dota_hazed'
    )
    