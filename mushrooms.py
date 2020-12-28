import socket
import urllib
import os
import time
import json
import requests
import wget
base_dir = './images/'

urla = 'https://mushroomobserver.org/api/images?content_type=jpg&name=amanita+muscaria&format=json'
imagurl='https://mushroomobserver.org/images/orig/627.jpg'
base_image_url='https://mushroomobserver.org/images/1280/'
base_req_url = 'https://mushroomobserver.org/api/images?content_type=jpg&confidence=0.1&format=json&include_synonyms=true&name='
mushroom_list = ['amanita+muscaria',
'galerina+marginata',
'panaeolus+cinctulus',
'panaeolina+foenisecii','gymnopilus+luteofolius','panaeolus+fimicola','amanita+ocreata','panaeolus+cyanescens',
'cortinarius','agaricus','panaeolus+papilionaceus',
'ganoderma+applanatum','abortiporus+biennis',
'aleuria+aurantia','gymnopilus+aeruginosus','gymnopilus+luteus','entoloma','armillaria+tabescens','lichen','morchella','russula','pleurotus+ostreatus','cantharellus+cinnabarinus','armillaria',
'coprinus+comatus','polyporales','panaeolus','schizophyllum+commune','lepista+nuda','dacrymyces+chrysospermus',
'agaricus+augustus','coprinellus+micaceus','Cortinarius violaceus','Stereum complicatum','Trametes gibbosa',
'Deconica coprophila',
'Clitocybe','Laetiporus sulphureus','Galerina','Amanita jacksonii','Agaricus campestris',
'Gymnopus','Tricholoma','Fomitopsis mounceae','Agaricomycetes','Trametes lactinea','Hydnellum peckii',
'Amanita calyptroderma','Ganoderma sessile','Lacrymaria lacrymabunda','Ganoderma oregonense','Pholiota','Marasmius',
'Calvatia cyathiformis','Agrocybe','Hebeloma','Daedaleopsis confragosa','Tricholomopsis rutilans','Lichenomphalia umbellifera',
'Amanita bisporigera','Lycoperdon perlatum','Amanita flavoconia','Boletaceae','Grifola frondosa','Gymnopus dryophilus','Tubaria furfuracea','Ischnoderma resinosum','amanita+pantherina','boletus+edulis','armillaria+mellea','boletus+betulicola','boletus+pinicola','cantharellus+cibarius','lactarius+deterrimus','lactarius+deliciosus','leccinum+scabrum','leccinum+melaneum','leccinum+variicolor','leccinum+versipelle','leccinum+vulpinum','russula+aeruginea','russula+claroflava','russula+vesca','russula+xerampelina','suillus+grevillei','suillus+luteus','suillus+granulatus',
'tricholoma+scalpturatum','xerocomellus+chrysenteron','mycena+galericulata']

new_list = ['ganoderma+pfeifferi',
           'coprinellus+micaceus',
           'fomitopsis+pinicola',
           'trametes+versicolor',
           'fomes+fomentarius',
           'psathyrella+candolleana']

def fetch_images(mush_name):
    res = requests.get(base_req_url+mush_name).json()
    return res['results']

def get_images_to_folder(folder_name,images_list):
    for image_id in images_list:
        url = base_image_url+str(image_id)+'.jpg'
        file_name = base_dir+folder_name+'/'+str(image_id)+'.jpg'
        time.sleep(3.1)
        try:
            wget.download(url,file_name)
        except:
            print(file_name)
            print('Some error occured! on {}'.format(folder_name))

for mush_name in new_list:
    mush_name = mush_name.replace(' ','+')
    mush_name = mush_name.lower()
    try:
        os.makedirs(base_dir+mush_name)
    except:
        print('{} aready exist'.format(mush_name))
    images_ids = fetch_images(mush_name)
    get_images_to_folder(mush_name,images_ids)
