import os
from PIL import Image
from .models import Post
from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageUploadForm
from .models import ImageClassifier,Post
import numpy as np
import glob
model_path = 'star/team3_new.h5'  # 실제 모델 파일의 경로로 수정
            # 이미지 분류기 초기화
classifier = ImageClassifier(model_path)
def classify_image(request):

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            # 모델 경로 설정

            # 이미지 분류
            result = classifier.classify_image(image)
            Post.objects.create(confidence=result['confidence'], result=result['class_label'])
            return render(request, 'result.html', {'result': result})
    else:
        form = ImageUploadForm()
        
    return render(request, 'index.html', {'form': form})

def rank(request):
    results = Post.objects.all()
    return render(request, 'ranking.html', {'results': results})

def sub_menu_1(request):
    return render(request, 'sub_menu_1.html')
