#Librerías necesarias
import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
from os.path import exists # Verifica la existencia de archivos
from scipy import stats as st


# Funciones parciales 
## Preprocesamiento
def filtroErosionesDilataciones(img,n_erosiones,n_dilataciones):
    ''' 
    img: Imagen en RGB
    n_erosiones: Número de erosiones
    n_dilataciones: Número de dilataciones

    '''
    elemento_rectangular = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel=elemento_rectangular
    
    opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = 1)
    closing= cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 1)

    
    
    img_erosion= cv2.erode(closing,kernel,iterations = n_erosiones)

    img_erosion_dilatacion=cv2.dilate(img_erosion,kernel,\
    iterations = n_dilataciones)

    return img_erosion_dilatacion

def filtroApertura(img,n_aperturas):
    ''' 
    img: Imagen en RGB
    n_aperturas: Número de Aperturas
    '''
    elemento_rectangular = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel=elemento_rectangular
    
    opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = 1)
    closing= cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 1)

    opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel, iterations = n_aperturas)
    return opening

def filtroCierre(img,n_cierres):
    ''' 
    img: Imagen en RGB
    n_cierres: Número de Cierres
    '''
    elemento_rectangular = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel=elemento_rectangular
    
    opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = 1)
    closing= cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 1)

    closing = cv2.morphologyEx(closing,cv2.MORPH_CLOSE,kernel, iterations = n_cierres)
    return closing

def filtradoEnColor(image_blur):
    # Rango de color para eliminar el verde
    light_green = (40, 40, 40)
    dark_green = (70, 255, 255)

    # Rango de color para eliminar la luz [rrr2]

    light_white = (0, 0, 231)
    dark_white = (180, 18, 255)

    # Filtrado en verde
    # conversion a HSV (matiz, saturación, valor)
    image_blur_HSV= cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image_blur_HSV, light_green, dark_green)
    result = cv2.bitwise_and(image_blur, image_blur, mask=mask)
    resultadoSinVerde=cv2.subtract(image_blur,result)

    # Filtrado en blanco
    mask = cv2.inRange(image_blur_HSV, light_white, dark_white)
    result = cv2.bitwise_and(image_blur, image_blur, mask=mask)
    resultadoSinBlanco=cv2.subtract(resultadoSinVerde,result)

    return resultadoSinBlanco

def posMax(Vector):
    ''' 
    Calcula la posición del máximo componete del Vector
    Vector: nx1 vector para hallar el máximo
    '''
    return np.where(Vector == np.amax(Vector))[0][0]

def myThreshold(img_rbg):
    # Conversión a escala de grises
    gray=cv2.cvtColor(img_rbg, cv2.COLOR_RGB2GRAY) 

    # Umbralización 'adaptativa' con THRESH_OTSU
    # img_thr_THRESH_OTSU= cv2.threshold(img_gray_blur[i],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # Umbralización con threshold
    
    return cv2.threshold(gray,3,255,0)[1] 

def unosDentroMascara(mask,cnt):

    # https://stackoverflow.com/questions/50670326/how-to-check-if-point-is-placed-inside-contour
    # https://docs.opencv.org/4.5.5/dc/d48/tutorial_point_polygon_test.html
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            dist = cv2.pointPolygonTest(cnt,(j,i),False)
            if dist>=0:
                mask[i,j]=1

    return mask

def mascaraWatershed(markers):
     n,m=markers.shape
     auxMarkers=np.array([],dtype=int)

     for i in range(n):
          for j in range (m):
               if(markers[i,j]!=-1 and markers[i,j]!=1):
                    auxMarkers=np.append(auxMarkers,markers[i,j])


     moda=st.mode(auxMarkers)[0][0]
     auxMarkers=np.zeros((n,m),dtype=np.uint8)
     
     for i in range(n):
          for j in range (m):
               if(markers[i,j]==moda):
                    auxMarkers[i,j]=255 

     return auxMarkers

# función principal
def esegmentacion(imagenes,rutaAndNombre,formato):

    # arreglos
    image_blur = []
    image_colorFilter= []
    image_grayFilter= []
    image_filter=[]
    lisContours=[]
    image_minAreaRect=[] # Array en el que sobreescribe el contorno
    image_umbralizada=[]
    image_segmentada=[]
    imagen_segmentada_watershep=[]
    imagenes_finales=[]

    # Variables auxiliares
    imagenesFinales=[]

    for i in range(len(imagenes)):
        # Filtro GaussianBlur
        image_blur.append(cv2.GaussianBlur(imagenes[i],(5,5),0))
        

        image_colorFilter.append(filtradoEnColor(image_blur[i]))
        # n_erosiones=math.ceil(math.log(closing.shape[0]*closing.shape[1]))
        # n_erosiones=7
        # n_dilataciones=7
        n_aperturas=15
        # n_cierres=15
    

        
        #image_filter.append(filtroErosionesDilataciones(image_colorFilter[i],\
        #    n_erosiones,n_dilataciones))
        

        image_grayFilter.append(cv2.cvtColor(image_colorFilter[i], cv2.COLOR_RGB2GRAY))

        image_filter.append(filtroApertura(image_grayFilter[i],n_aperturas))
        #image_filter.append(filtroCierre(image_colorFilter[i],n_cierres))

    
    contador=0
    i=0
    while contador<len(imagenes):

        #image_umbralizada.append(myThreshold(image_filter[i]))
        #image_umbralizada.append(cv2.threshold(image_filter[i],3,255,0)[1])
        aux_image_umbralizada=cv2.threshold(image_filter[contador],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        lisContours.append(cv2.findContours(\
                aux_image_umbralizada,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0])
        # https://docs.opencv.org/4.5.5/dd/d49/tutorial_py_contour_features.html
        # https://docs.opencv.org/4.5.5/d6/d6e/group__imgproc__draw.html#ga57be400d8eff22fb946ae90c8e7441f9

        
        
        areas=np.array([],float)
        
        for contorno in lisContours[contador]:        
            areas=np.append(areas,cv2.contourArea(contorno))

        

        try:

            ## Se elige la máxima área
            posicionMaximaArea=posMax(areas)
            rect = cv2.minAreaRect(lisContours[contador][posicionMaximaArea])
            box = cv2.boxPoints(rect)
            box = np.int0(box)



        except ValueError: 
            print('No se logró sacar el contorno')
            contador+=1

        else:
            
            

            if len(lisContours[contador][posicionMaximaArea]) >= 5:
                
                image_umbralizada.append(aux_image_umbralizada)

                image_minAreaRect.append(imagenes[contador].copy())

                cv2.drawContours(image_minAreaRect[i], contours=[box],\
                contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)


                ellipse = cv2.fitEllipse(lisContours[contador][posicionMaximaArea])
                cv2.ellipse(img=image_minAreaRect[i],box=ellipse,color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
                
                #https://itecnote.com/tecnote/python-get-mask-from-contour-with-opencv/
                
                # Se realiza la máscara
                n=image_minAreaRect[i].shape[0]
                m=image_minAreaRect[i].shape[1]
                # image_mascaras.append(np.zeros((n,m), np.uint8))
                # image_mascaras[i]=unosDentroMascara(image_mascaras[i],box)

                mask=np.zeros((n,m), np.uint8)
                mask=unosDentroMascara(mask,box)


                #cv2.ellipse(img=image_mascaras[i],box=ellipse,color=(255), thickness=2, lineType=cv2.LINE_AA)
                
                # cv2.drawContours(image_minAreaRect[i], contours=[box],\
                # contourIdx=0, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

                # Se discrimina el área de interés
                auxImagenSegmentada=cv2.bitwise_and(image_blur[contador], image_blur[contador], mask=mask)
                #auxImagenSegmentada=filtradoEnColor(auxImagenSegmentada)
                #auxImagenSegmentada=filtroApertura(auxImagenSegmentada,3)
                image_segmentada.append(auxImagenSegmentada)

                # Imágenes que se les pudo sacar el contorno
                imagenesFinales.append(contador)
                i+=1
            
            contador+=1 

    for i in range(len(image_umbralizada)):
        # Finding sure foreground area
        #i=5
        kernel = np.ones((3,3),np.uint8)
        dist_transform = cv2.distanceTransform(image_umbralizada[i],cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # sure background area
        sure_bg = cv2.dilate(image_umbralizada[i],kernel,iterations=3)


        # Conversión del tipo de datos
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0



        image_watershed=image_segmentada[i].copy()
        img=image_watershed.copy()

        markers = cv2.watershed(image_watershed,markers)
        img[markers == -1] = [0,0,0]


        mascaraWatershedFinal=mascaraWatershed(markers)

        imagen_segmentada_watershep.append(cv2.bitwise_and(image_segmentada[i], image_segmentada[i], mask=mascaraWatershedFinal))
    
        imagenes_finales.append(imagenes[imagenesFinales[i]])

        # Zona de guardado
        #print(f'{rutaAndNombre}{i}.{formato}')
        cv2.imwrite(f'{rutaAndNombre}{i}Original.{formato}', cv2.cvtColor(imagenes_finales[i],cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{rutaAndNombre}{i}.{formato}', cv2.cvtColor(imagen_segmentada_watershep[i],cv2.COLOR_RGB2BGR))

    return [imagenes_finales,image_segmentada,imagen_segmentada_watershep]


ruta='../../figs/baseDatos/'

# Pixeles del mínimo entre el ancho y el alto
minSize=256


# Importación de monilia
n_imag_monilia=105
# n_imag_monilia=5
imag_monilia=[]

# Importacion de Phytophthora 

n_imag_fito=106
# n_imag_fito=5
imag_fito=[]

# Importación de cacao saludable

n_imag_healty=100
# n_imag_healty=5
imag_healty=[]

# Variables para guardar 

# rutaGuardarFito='../../figs/imagenesSegmentadasAutomaticamente/Fito/Fito'
# rutaGuardarMonilia='../../figs/imagenesSegmentadasAutomaticamente/Monilia/Monilia'
# rutaGuardarSaludable='../../figs/imagenesSegmentadasAutomaticamente/Sana/Sana'

rutaGuardarFito='../../figs/imagenesSegmentadasAutomaticamenteTotales/Fito/Fito'
rutaGuardarMonilia='../../figs/imagenesSegmentadasAutomaticamenteTotales/Monilia/Monilia'
rutaGuardarSaludable='../../figs/imagenesSegmentadasAutomaticamenteTotales/Sana/Sana'


for i in range(n_imag_monilia):
    if exists(f'{ruta}Monilia/Monilia{i+1}.jpg'):
        imag_aux = cv2.imread(f'{ruta}Monilia/Monilia{i+1}.jpg')
        imag_aux=cv2.cvtColor(imag_aux,cv2.COLOR_BGR2RGB)
        
        n=imag_aux.shape[0]
        m=imag_aux.shape[1]

        escala= minSize/min(n,m)
        imag_aux= cv2.resize(imag_aux, None, fx=escala, fy= escala,\
                                    interpolation= cv2.INTER_LINEAR)

        imag_monilia.append(imag_aux)

for i in range(n_imag_fito):
    if exists(f'{ruta}Fito/Fito{i+1}.jpg'):
        imag_aux = cv2.imread(f'{ruta}Fito/Fito{i+1}.jpg')
        imag_aux=cv2.cvtColor(imag_aux,cv2.COLOR_BGR2RGB)
        
        n=imag_aux.shape[0]
        m=imag_aux.shape[1]

        escala= minSize/min(n,m)
        imag_aux= cv2.resize(imag_aux, None, fx=escala, fy= escala,\
                                    interpolation= cv2.INTER_LINEAR)

        imag_fito.append(imag_aux)


for i in range(n_imag_healty):
    if exists(f'{ruta}Sana/Sana{i+1}.jpg'):
        imag_aux = cv2.imread(f'{ruta}Sana/Sana{i+1}.jpg')

        imag_aux=cv2.cvtColor(imag_aux,cv2.COLOR_BGR2RGB)
        
        n=imag_aux.shape[0]
        m=imag_aux.shape[1]

        escala= minSize/min(n,m)
        imag_aux= cv2.resize(imag_aux, None, fx=escala, fy= escala,\
                                    interpolation= cv2.INTER_LINEAR)

        imag_healty.append(imag_aux)



esegmentacion(imag_monilia,rutaGuardarMonilia,'jpg')
esegmentacion(imag_fito,rutaGuardarFito,'jpg')
esegmentacion(imag_healty,rutaGuardarSaludable,'jpg')
