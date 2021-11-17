## Pliki HDF5

W pliku HDF5 znajdują się dane dla każdego fotonu w takiej kolejności

[E,X,Y,angle1,angle2,dX,dY,dZ]

oraz dla każdej z powyższych zmiennych wartość średnie tej zmiennej i odchylenie standardowe.


## Struktura katalogów i plików oraz zakodowane w ich nazwach parametry

W nazwach katalogów i znajdujących się w nich plikach HDF5 zakodowane są parametry wiązki w następujący sposób:

workPath + 'DISP_' + D + '_ANGLE_' + A + '/HDF5/' + E + s + 'TrainGenerator.hdf5


Wiązka elektronowa PRIMO jest zadana czterema parametrami:
- szerokość połówkowa widma energetycznego (D)
- rozbiezność kątowa (A)
- energia nominalna (E)
- spot size (s)

Rozbieżność kątowa i szerokość połówkowa widma energetycznego jest kodowana w nazwach katalogów (np. DISP_1_ANGLE_3 oznacza szerokośc połówkową 1MeV i rozbiezność kątową 3 stopnie).
Energia nominalna i spot size są kodowane w nazwach przestrzeni fazowych: litery a, b, c, d i e oznaczają energie nominalne 5.6, 5.8, 6.0, 6.2 i 6.4 MV, a cyfry 1, 2, 3, 4 i 5 oznaczają spot size 0, 1, 2, 3 i 4 mm.


D: ['0','0.5','1'] # szerokość połówkowa widma energetycznego [MeV]
A: ['0','1','2','3'] # rozbieżność kątowa


eDict = {'a': 5.6,'b':5.8,'c':6.0,'d':6.2,'e':6.4} # energia nominalna [MV]
sDict = {'1': 0,'2':0.1,'3':0.2,'4':0.3,'5':0.4} # spot size [mm]


Np. przestrzeń fazowa a1TrainGenerator.hdf5 z katalogu DISP_0.5_ANGLE_0 odpowiada:
szerokość połówkowa widma energetycznego 0.5MeV, rozbieżności kątowej 0 stopni, energii nominalnej 5.6MV i spot size 0mm.


W sumie na Prometeuszu jest 12 folderów:

DISP_0.5_ANGLE_0  DISP_0.5_ANGLE_2  DISP_0_ANGLE_0  DISP_0_ANGLE_2  DISP_1_ANGLE_0  DISP_1_ANGLE_2  
DISP_0.5_ANGLE_1  DISP_0.5_ANGLE_3  DISP_0_ANGLE_1  DISP_0_ANGLE_3  DISP_1_ANGLE_1  DISP_1_ANGLE_3

A w każdym  25 plików hdf5:

a1TrainGenerator.hdf5  a5TrainGenerator.hdf5  b4TrainGenerator.hdf5  c3TrainGenerator.hdf5  d2TrainGenerator.hdf5  e1TrainGenerator.hdf5  e5TrainGenerator.hdf5
a2TrainGenerator.hdf5  b1TrainGenerator.hdf5  b5TrainGenerator.hdf5  c4TrainGenerator.hdf5  d3TrainGenerator.hdf5  e2TrainGenerator.hdf5
a3TrainGenerator.hdf5  b2TrainGenerator.hdf5  c1TrainGenerator.hdf5  c5TrainGenerator.hdf5  d4TrainGenerator.hdf5  e3TrainGenerator.hdf5
a4TrainGenerator.hdf5  b3TrainGenerator.hdf5  c2TrainGenerator.hdf5  d1TrainGenerator.hdf5  d5TrainGenerator.hdf5  e4TrainGenerator.hdf5


Na lhcbgpu pobrany jest 1 przykładowy folder (DISP_0_ANGLE_0/HDF5) wraz z plikami HDF5 wewnątrz niego.


## Generator
Skrypt HDF5Generator.py zawiera generator cząstek (losuje paczki danych, które podawane są na wejście do treningu np. GANa).

W zależności od argumentu normalize = True/False)
Generator zwraca znormalizowane lub surowe dane dla każdej cząstki w takiej kolejności, jaka jest wymagana przez GAN Francuzów tzn. [E,X,Y,dX,dY,dZ].