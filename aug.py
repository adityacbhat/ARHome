import cv2
import random
import os
def data_augmenter(path,num_ofsample_perImage,class_ids,sample_store_path):
    def convert_labels(img,x1, y1, x2, y2):
        def sorting(l1, l2):
            if l1 > l2:
                lmax, lmin = l1, l2
                return lmax, lmin
            else:
                lmax, lmin = l2, l1
                return lmax, lmin
        size = img.shape
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        dw = 1./size[1]
        dh = 1./size[0]
        x = (x2 + x1)/2.0
        y = (y2 + y1)/2.0
        w = x2 - x1
        h = y2 - y1
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return x,y,w,h
    
    #Finds the labels
    def label_finder(class_ids):
        lbl=[]
        for all_labels in os.listdir(path+'labels/'):
            f1=open(path+'labels/'+all_labels,'r')
            lines=f1.readlines()
            for stuff in lines:
                z=stuff.split(' ')
                z[0]=int(z[0])
                if(z[0] in class_ids):
                    lbl.append(all_labels)
                    break
            f1.close()
        print("Total number of images with samples: ",len(lbl))
        return lbl

    def sample_collector(class_ids):
        class_dict={}
        thresh_h=thresh_w=35
        try:
            os.mkdir(sample_store_path+'samples1')
            os.mkdir(sample_store_path+'augmented_images')
        except:
            pass
        for contents in class_ids:
            class_dict[contents]=0
        label_with_samples=label_finder(class_ids)
        for samples in label_with_samples:
            f1=open(path+'labels/'+samples,'r')
            lines=f1.readlines()
            try:
                name=samples.split('.')[0]+'.jpeg'
                img=cv2.imread(path+'images/'+name)
                img_h, img_w, _ = img.shape
            except: 
                name=samples.split('.')[0]+'.jpg'
                img=cv2.imread(path+'images/'+name)
                img_h, img_w, _ = img.shape

            for stuff in lines:
                z=stuff.split(' ')
                z[0]=int(z[0])
                if(z[0] in class_ids):
                    x,y,w,h=z[1],z[2],z[3],z[4]

                    x1 = float(img_w) * (2.0 * float(x) - float(w)) / 2.0
                    y1 = float(img_h) * (2.0 * float(y) - float(h)) / 2.0
                    x2 = float(img_w) * (2.0 * float(x) + float(w)) / 2.0
                    y2 = float(img_h) * (2.0 * float(y) + float(h)) / 2.0
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                    temp=img[y1:y2,x1:x2]
                    temp_img_h, temp_img_w, _ = temp.shape
                    if(temp_img_h>=thresh_h or temp_img_w>=thresh_w):


                        cv2.imwrite(os.path.join(sample_store_path+'samples1','classID'+str(z[0])+'_sampleNum'+str(class_dict[z[0]])+'.jpg'),temp)
                        class_dict[z[0]]+=1
        print("Samples of individual class: Total Number of samples\n",class_dict)
        for cl in class_dict.items():
            if(cl[1]==0):
                print(" 0 samples collect for class: ",cl[0],'Change thresh val')
                break
        return
        
        
        
    sample_collector(class_ids)
    
    
    def samplepaster(sample_store_path,class_ids,path,num_ofsample_perImage):
        
        class_dict={}
        for contents in class_ids:
                class_dict[contents]=[]
        for i in os.listdir(sample_store_path+'samples1'):
          #  print("image ",i)
            img=cv2.imread(sample_store_path+'samples1/'+i)
            class_dict[int(i.split('_')[0][::-1][0])].append(img)
        from tqdm import tqdm 
        from scipy import ndimage
        

        loop=tqdm(total=len(os.listdir(path+'labels/')),position=0,leave=False)


        for k,i in enumerate(os.listdir(path+'labels/')) :
            f1=open(path+'labels/'+i,'r')
            lines=f1.readlines()
            loop.set_description("Augmenting.....",k)
            loop.update(1)


            try:
                name=i.split('.')[0]+'.jpeg'
                

                img=cv2.imread(path+'images/'+name)
                img_h, img_w, _ = img.shape    
            except:
                name=i.split('.')[0]+'.jpg'
                

                img=cv2.imread(path+'images/'+name)
                img_h, img_w, _ = img.shape  

            ann=[0]

            for con in range(len(class_dict)):
                val=0
                while(val<num_ofsample_perImage):
                    while(True):
                        x_offset=random.randint(10,img_w-200)
                        y_offset=random.randint(10,img_h-200)

                        closex=ann[min(range(len(ann)), key = lambda xstuff: abs(ann[xstuff]-x_offset))]
                        closey=ann[min(range(len(ann)), key = lambda ystuff: abs(ann[ystuff]-y_offset))]


                        if(abs(closex-x_offset)>25 and abs(closey-y_offset)>25 ): 
                            ann.append(x_offset)
                            ann.append(y_offset)
                            break
                       
                    rval=random.randint(0,len(class_dict[list(class_dict.keys())[con]])-1)
                   
                    
            
                    temp=class_dict[list(class_dict.keys())[con]][rval]
                   
                    #Randomly flipping along Y-axis:
                    randomised=random.choice([True,False])
                    if(randomised):
                        flipped=cv2.flip(temp,1)
                    else:
                        flipped=temp

                    
                        
                    rotated = ndimage.rotate(flipped, random.randint(-15,15))
               
                    if(img.shape[0]>rotated.shape[0] and img.shape[1]>rotated.shape[1] ):
             
                        img[y_offset:y_offset+rotated.shape[0], x_offset:x_offset+rotated.shape[1]] = rotated
                       

                        val+=1


                        yot=y_offset
                        yo_temp=y_offset+rotated.shape[0]
                        xot=x_offset
                        xo_temp=x_offset+rotated.shape[1]

                        yolo_res=convert_labels(img,xot,yot,xo_temp,yo_temp)

                        f2=open(path+'labels/'+i,'a')

                        f2.write('\n'+str(list(class_dict.keys())[con])+' ')


                        if(len(yolo_res)>0):
                            for cont in yolo_res:
                                f2.write(str(cont)+' ')

                        f2.close()

                    else:
                            continue
            alpha =1 # Contrast control (1.0-3.0)
            beta = random.randint(5,55) # Brightness control (0-100)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
                
            cv2.imwrite(os.path.join(sample_store_path+'augmented_images/',i.split('.')[0])+'.jpg',img)

            f1.close()   
    
        print("\nDone")
    print("\n Saved at 'sample1 folder'. Check for Image Quality and Size. Modify 'thresh_w' or 'thresh_w' values for better/More samples ")
    confirm=input("ANNOTATIONS FILES WILL BE APPENDED WITH AUGMENTED SAMPLES Continue(Y/N): ")
    if(confirm=='Y' or confirm=='y'):
        samplepaster(sample_store_path,class_ids,path,num_ofsample_perImage)
    else:
       
        return


def classification_augmenter(original_path,aug_samples_path,output_path):
    aug_samples={}
    for i in os.listdir(aug_samples_path):
        img=cv2.imread(aug_samples_path+i)
        aug_samples[i]=img
        
    from tqdm import tqdm 
        
        

    loop=tqdm(total=len(os.listdir(original_path)),position=0,leave=False)
    
    for k,norm in enumerate(os.listdir(original_path)):
        loop.set_description("Augmenting.....",k)
        loop.update(1)
        img1=cv2.imread(original_path+norm)
        img_h,img_w,_=img1.shape

        img=cv2.imread(aug_samples_path+random.choice(list(aug_samples)))
        img=cv2.resize(img,(img_w,img_h))
        
        #flipping:
        r=random.choice([0,1])
        img=cv2.flip(img,r)
        
        alpha=random.randrange(75,100,1)/100
        
        img1 = cv2.addWeighted(img1, 1, img, 0.5, 1)
        
        #rotating
        rotate_code=random.choice([cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180])
       
        img1=cv2.rotate(img1,rotate_code)
        if('Augmented_data' not in os.listdir(output_path)):
            os.mkdir(output_path+'Augmented_data')
        cv2.imwrite(os.path.join(output_path+'Augmented_data',norm),img1)
    print("\nDone")
    #cv2.imshow('frme',cv2.resize(img1,(1200,600)))
   # cv2.waitKey(0)
    #cv2.destroyAllWindows()