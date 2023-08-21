<img src="https://capsule-render.vercel.app/api?type=rounded&color=auto&height=200&section=header&text=⭐닮은꼴%20추출기⭐&fontSize=70" /> 


## 서비스 간단 소개

> **CNN 기법**을 사용해 사용자가 사진을 입력하면 사전에 모델이 학습한 연예인 범주 중 하나를 출력해주는 서비스

<br>
<br>

## 프로젝트 개요

-  **프로젝트 명칭** : 닮은꼴 추출기
-  **개발 인원** : 6명
-  **개발 기간** : 2023.05.15 ~ 2023.05.19
-  **개발 언어** : Python
-  **개발 환경** : Django, WSL
  <br>

<br>

## 모델 선정

> 모델은 사전학습된 **VGG16 모델**을 가져와서 전이학습시키는 방식으로 진행.

<br>
<br>
<br>

## 1. 데이터 수집

> 구글에 있는 연예인 이미지를 연예인 1명당 100장씩 크롤링하여 수집.

<br>

## 2. 데이터 전처리

> 크롤링한 이미지 데이터 중 인물 사진이 아닌 이미지 제거
> 해당 연예인이 아닌 이미지 제거
> VGG16 모델의 최대 성능을 뽑기 위해 224*224 크기로 이미지 크기 수정

<br>

## 3. 이미지 보강하기

> 전처리 과정에서 많은 이미지 데이터들이 소실되었기 때문에 이미지 보강을 해주었음

<br>

## 4. 배포하기

> AWS에서 인스턴스를 생성하여 배포함.

<br>
<br>

## 서비스 화면

<img src="https://postfiles.pstatic.net/MjAyMzA4MjFfMjY0/MDAxNjkyNTg4MDA0MTI0.EATGrNVlbOpqUAwbcysr52qyHa-2bNscrQdNAajUrbMg.8VSqDrVA-G6gVuWABaifArK3BFSG0E_UmMVIm8_6hN4g.PNG.tbtgmltn97/image.png?type=w773"/>
<img src="https://postfiles.pstatic.net/MjAyMzA4MjFfMjgg/MDAxNjkyNTg4MDM2MDM1.orGcMx7NB-0m-1XlqHb2u9EOjqBiBlw_dvGRXkSXv1cg.xv7fEjOa17Xd3njcuIu6UenDwhUHFzhAzBj_xFtbXzwg.PNG.tbtgmltn97/image.png?type=w773"/>
<br>

<br>
