# Create your views here.
import os

import tensorflow as tf
from Web.models import IMG
from django.shortcuts import render
from Web.CNN_PREDICT import predict
import json


# Create your views here.
def uploadImg(request):
    if request.method == 'POST':
        new_img = IMG(
            img=request.FILES.get('img')
        )
        new_img.save()
    return render(request, 'uploadimg.html')
    # return render(request, 'picture_angular_form.html')


def showImg(request):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            myimg = IMG.objects.all()
            # for i in myimg:
            #     print("图片的路径：" + i.img.url)
            # print("最后一张图片的路径是：" + myimg[len(myimg) - 1].img.url)
            image = myimg[len(myimg) - 1]
            content = {
                # "image": image
            }
            module_dir = os.path.dirname(__file__)[:-3]  # get current directory
            i_path = image.img.url[14:]
            print("图片名称：" + i_path)
            img_path = os.path.join(module_dir, 'media', 'upload', i_path)
            result = predict(img_path)
            # for index in range(len(result)):
            #     print('概率:', result[index])
            content['result'] = result
    return render(request, 'showimg.html', {'content': json.dumps(content)})
