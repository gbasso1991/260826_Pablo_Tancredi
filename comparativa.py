#%% Librerias y paquetes
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from scipy.interpolate import interp1d
from clase_resultados import ResultadosESAR
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))

                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor

                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer SAR, tau y Hc de resultados
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    Hc = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m'])
    return SAR, tau, Hc
#%% funcion banda temperatura
def banda_temperatura(t, T, N=500, kind='linear'):
    """
    Interpola varias curvas T(t) sobre una grilla temporal común y
    calcula estadísticas punto a punto.

    Parameters
    ----------
    t : list of np.ndarray
        Lista de vectores de tiempo.
    T : list of np.ndarray
        Lista de vectores de temperatura.
    N : int, optional
        Número de puntos de la grilla común.
    kind : str, optional
        Tipo de interpolación (interp1d).

    Returns
    -------
    tt : list of np.ndarray
        Lista original de tiempos.
    TT : list of np.ndarray
        Lista original de temperaturas.
    t_common : np.ndarray
        Grilla temporal común.
    Tmin : np.ndarray
        Temperatura mínima en cada instante.
    Tmax : np.ndarray
        Temperatura máxima en cada instante.
    Tmean : np.ndarray
        Temperatura promedio en cada instante.
    """

    # intervalo temporal común
    tmin = max(tt.min() for tt in t)
    tmax = min(tt.max() for tt in t)

    t_common = np.linspace(tmin, tmax, N)

    # interpolación
    Ti = []
    for tt, TT in zip(t, T):
        f = interp1d(tt, TT, kind=kind)
        Ti.append(f(t_common))

    Ti = np.asarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%% 1- 260706_NF-2P2C
nombre_2P2C='NF@2PAA2cit 260717AUV'
ciclos_2P2C = glob("**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_2P2C = glob("**/*resultados.txt", recursive=True)
ciclos_2P2C.sort()
resultados_2P2C.sort()
conc_2P2C =  13 #g/L (fotom g3m)

print('Importando ciclos de', nombre_2P2C,'\n')
for p in ciclos_2P2C:
    print('  ',p)
print('\n')
for res in resultados_2P2C:
    print('  ',res)
print('-'*50)

SAR_2P2C, tau_2P2C, Hc_2P2C = extraer_SAR_tau(resultados_2P2C)
res_2P2C=[]
#%% ploteo ciclos
fig00, axs =plt.subplots(1,2,figsize=(10,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('57 kA/m',loc='left')

for i,e in enumerate(ciclos_2P2C):
    if '100dA' in e:
        _,_,_, H_2P2C,M_2P2C,_ = lector_ciclos(ciclos_2P2C[i])
        print(e)
        axs[0].plot(H_2P2C/1000,M_2P2C,'-',label=f'{SAR_2P2C[i]:.3uS}')

for i,e in enumerate(ciclos_2P2C):
    if '150dA' in e:
        _,_,_, H_2P2C,M_2P2C,_ = lector_ciclos(ciclos_2P2C[i])
        print(e)
        axs[1].plot(H_2P2C/1000,M_2P2C,'-',label=f'{SAR_2P2C[i]:.3uS}')

for a in axs:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio {nombre_2P2C} \n300 kHz & [38, 57] kA/m\nC = {conc_2P2C:.1f} g/L')


fig000, axs =plt.subplots(1,1,figsize=(9,7),constrained_layout=True,sharey=True,sharex=True)
axs.set_ylabel('M (A/m)')
ls=['-','--','-.']*3

for i,e in enumerate(ciclos_2P2C):
    if '100dA' in e:
        _,_,_, H_2P2C,M_2P2C,_ = lector_ciclos(ciclos_2P2C[i])
        print(e)
        axs.plot(H_2P2C/1000,M_2P2C,'-',c='C1',ls=ls[i],label=f'{SAR_2P2C[i]:.3uS}')

for i,e in enumerate(ciclos_2P2C):
    if '150dA' in e:
        _,_,_, H_2P2C,M_2P2C,_ = lector_ciclos(ciclos_2P2C[i])
        print(e)
        axs.plot(H_2P2C/1000,M_2P2C,'-',c='C2',ls=ls[i],label=f'{SAR_2P2C[i]:.3uS}')


axs.grid()
axs.set_xlabel('H (kA/m)')
axs.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)',ncol=2)
plt.suptitle(f'Ciclos promedio {nombre_2P2C} \n300 kHz & [38, 57] kA/m\nC = {conc_2P2C:.1f} g/L')

#%%
print('Resultados 2P2C', '='*80,'\n')
for r in resultados_2P2C:
    res_2P2C.append(ResultadosESAR(os.path.dirname(r)))
rates_2P2C = []

#%% Templogs
fig01, axs =plt.subplots(1,2,figsize=(10,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('57 kA/m',loc='left')


for i,r in enumerate(res_2P2C):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_2P2C.append(rate)
    print(f'WRate = {rate:.2f} °C/s')
    if i<3:
        axs[0].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f}')
    elif i<6:
        axs[1].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f}')
    else:
        axs[2].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f}')

axs[0].set_ylabel('T (°C)')
for a in axs:
    a.grid()
    a.set_xlabel('t (s)')
    a.legend(loc='upper left',frameon=True,shadow=True,title='Warming Rate (°C/s)',ncol=3)
plt.suptitle(f'Templogs {nombre_2P2C} \n300 kHz & [38, 57] kA/m\nC = {conc_2P2C:.1f} g/L')

################################################################################################################################
#%% Normalizo ciclos por concentracion y ploteo comparativo

fig2, axs =plt.subplots(1,2,figsize=(10,5),constrained_layout=True,sharey=False,sharex=False)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('57 kA/m',loc='left')

      
for i,e in enumerate(ciclos_2P2C):
    if '100dA' in e:
        _,_,_, H_2P2C,M_2P2C,_ = lector_ciclos(ciclos_2P2C[i])
        print(os.path.split(e)[-1])
        axs[0].plot(H_2P2C/1000,M_2P2C/conc_2P2C,'-',c='C0',label=f'NF@cit\n{conc_2P2C} g/L' if i==0 else "")

for i,e in enumerate(ciclos_2P2C):
    if '150dA' in e:
        _,_,_, H_2P2C,M_2P2C,_ = lector_ciclos(ciclos_2P2C[i])
        print(os.path.split(e)[-1])
        axs[1].plot(H_2P2C/1000,M_2P2C/conc_2P2C,'-',c='C0',label=f'NF@cit\n{conc_2P2C} g/L' if i==3 else "")

axs[0].set_ylabel('M/[NPM] (Am²/kg)')

for a in axs:
    a.set_xlabel('H (kA/m)')
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,ncol=2)
plt.suptitle(f'Ciclos promedio nomalizados por concentracion\n300 kHz & [38, 57] kA/m\n')

#%% ploteo comparativo de errorbars de ESAR
categorias = ['260717AUV\nNF@2PAA2cit']
x = np.arange(len(categorias))

fig3, (ax,ax2) = plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(SAR_2P2C[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for i,s in enumerate(SAR_2P2C[3:6]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('57 kA/m',loc='left')
for a in [ax,ax2]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
    

ax.set_ylabel('ESAR (W/g)')
plt.suptitle(f'ESAR\n300 kHz & [38, 57] kA/m\n')

plt.show()
#%% ploteo comparativo de tau
categorias = ['260717AUV\nNF@2PAA2cit']
x = np.arange(len(categorias))

fig4, (ax,ax2) = plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(tau_2P2C[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for i,s in enumerate(tau_2P2C[3:6]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('57 kA/m',loc='left')
for a in [ax,ax2]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
ax.set_ylabel('tau (ns)')
plt.suptitle(f'tau\n300 kHz & [38, 57] kA/m\n')
plt.show()

#%% Idem Hc
categorias = ['260717AUV\nNF@2PAA2cit']
x = np.arange(len(categorias))

fig5, (ax,ax2) = plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(Hc_2P2C[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for i,s in enumerate(Hc_2P2C[3:6]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('57 kA/m',loc='left')
for a in [ax,ax2]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
ax.set_ylabel('Hc (kA/m)')
plt.suptitle(f'Hc\n300 kHz & [38, 57] kA/m\n')
plt.show()


#%% Salvo todas las figuras
fig00.savefig('00_ciclos_promedio_NF2P2C_260717AUV.png',dpi=300)
fig000.savefig('01_ciclos_promedio_all_NF2P2C_260717AUV.png',dpi=300)
fig01.savefig('02_templogs_NF2P2C_260717AUV.png',dpi=300)
fig2.savefig('04_ciclos_promedio_comparativa.png',dpi=300)
fig3.savefig('05_ESAR_comparativa.png',dpi=300)
fig4.savefig('06_tau_comparativa.png',dpi=300)
fig5.savefig('07_Hc_comparativa.png',dpi=300)


# %%
