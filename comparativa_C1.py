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
#%% C1
nombre_C1='C1'
ciclos_C1 = glob("C1/**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_C1 = glob("C1/**/*resultados.txt", recursive=True)
ciclos_C1.sort()
resultados_C1.sort()
conc_C1 =  20.56 #g/L 

print('Importando ciclos de', nombre_C1,'\n')
for p in ciclos_C1:
    print('  ',p)
print('\n')
for res in resultados_C1:
    print('  ',res)
print('-'*50)

SAR_C1, tau_C1, Hc_C1 = extraer_SAR_tau(resultados_C1)
res_C1=[]
#%% ploteo ciclos
fig00, axs =plt.subplots(1,5,figsize=(18,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_title('20 kA/m',loc='left')
axs[1].set_title('29 kA/m',loc='left')
axs[2].set_title('38 kA/m',loc='left')
axs[3].set_title('47 kA/m',loc='left')
axs[4].set_title('57 kA/m',loc='left')

for i,e in enumerate(ciclos_C1):
    if '050dA' in e:
        _,_,_, H_C1,M_C1,_ = lector_ciclos(ciclos_C1[i])
        print('1',os.path.basename(e))
        axs[0].plot(H_C1/1000,M_C1,'-',label=f'{SAR_C1[i]:.2uS}')

    elif '075dA' in e:
        _,_,_, H_C1,M_C1,_ = lector_ciclos(ciclos_C1[i])
        print('1',os.path.basename(e))
        axs[1].plot(H_C1/1000,M_C1,'-',label=f'{SAR_C1[i]:.3uS}')

    elif '100dA' in e:
        _,_,_, H_C1,M_C1,_ = lector_ciclos(ciclos_C1[i])
        print('1',os.path.basename(e))
        axs[2].plot(H_C1/1000,M_C1,'-',label=f'{SAR_C1[i]:.2uS}')

    elif '125dA' in e:
        _,_,_, H_C1,M_C1,_ = lector_ciclos(ciclos_C1[i])
        print('1',os.path.basename(e))
        axs[3].plot(H_C1/1000,M_C1,'-',label=f'{SAR_C1[i]:.3uS}')

    elif '150dA' in e:
            _,_,_, H_C1,M_C1,_ = lector_ciclos(ciclos_C1[i])
            print('1',os.path.basename(e))
            axs[4].plot(H_C1/1000,M_C1,'-',label=f'{SAR_C1[i]:.3uS}')

axs[0].set_ylabel('M (A/m)')

for a in axs.ravel():
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
    a.set_xlabel('H (kA/m)')
plt.suptitle(f'{nombre_C1} ciclos promedio \nf = 300 kHz & H$_0$ = 20/29/38/4757 kA/m\nC = {conc_C1:.1f} g/L')

#%%
print('Resultados C1', '='*80,'\n')
for r in resultados_C1:
    res_C1.append(ResultadosESAR(os.path.dirname(r)))
rates_C1 = []

#%% Templogs

fig01, axs =plt.subplots(1,5,figsize=(18,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_title('20 kA/m',loc='left')
axs[1].set_title('29 kA/m',loc='left')
axs[2].set_title('38 kA/m',loc='left')
axs[3].set_title('47 kA/m',loc='left')
axs[4].set_title('57 kA/m',loc='left')

for i,r in enumerate(res_C1):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_C1.append(rate)
    print(i,f'WRate = {rate:.2f} °C/s')
    if '050' in r.directorio:
        axs[0].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f}')
    elif '075' in r.directorio:
        axs[1].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f}')
    elif '100' in r.directorio:
        axs[2].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f}')
    elif '125' in r.directorio:
        axs[3].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f}')
    elif '150' in r.directorio:
        axs[4].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f}')
        
axs[0].set_ylabel('T (°C)')

for a in axs.ravel():
    a.grid()
    a.legend(loc='best',frameon=True,shadow=True,title='Warming Rate (°C/s)',ncol=2)
    a.set_xlabel('t (s)')

plt.suptitle(f'Templogs {nombre_C1}\nf = 238/300 kHz & H$_0$ = 38/57 kA/m\nC = {conc_C1:.1f} g/L')

#%%##############################################################################################################################
# ESAR vs H0

H0 = [19.9, 29.2, 38.5,47.8,57.0,57.0,57.0]
ticks = [19.9, 29.2, 38.5,47.8,57.0]

fig02, (a,b,c) = plt.subplots(nrows=3, figsize=(10,8), sharex=True,constrained_layout=True)

a.set_title(f'ESAR vs H$_0$ {nombre_C1}',loc='left')
a.errorbar(x=H0,y=[s.n for s in SAR_C1], yerr=[s.s for s in SAR_C1], fmt='.',ls='-', label='C1', capsize=5)

b.set_title(f'tau vs H$_0$ {nombre_C1}',loc='left')
b.errorbar(x=H0,y=[t.n for t in tau_C1], yerr=[t.s for t in tau_C1], fmt='.',ls='-', label='C1', capsize=5)

c.set_title(f'Hc vs H$_0$ {nombre_C1}',loc='left')
c.errorbar(x=H0,y=[h.n for h in Hc_C1], yerr=[h.s for h in Hc_C1], fmt='.',ls='-', label='C1', capsize=5)


a.set_ylabel('ESAR (W/g)')
c.set_xlabel('H$_0$ (kA/m)')

for ax in [a,b,c]:
    ax.grid()
    ax.set_xticks(ticks)  
    ax.set_xticklabels([str(h) for h in ticks])  
    ax.legend(ncol=2, loc='best',frameon=True,shadow=True,title='f = 300 kHz')
plt.suptitle(f'Comparativa de resultados {nombre_C1}\nC = {conc_C1:.1f} g/L')
plt.show()

#%% tau vs H0
# fig, b = plt.subplots(nrows=1, figsize=(7,5), constrained_layout=True)
# b.errorbar(x=H0,y=[t.n for t in tau_C1], yerr=[t.s for t in tau_C1], fmt='.',ls='-', label='C1', capsize=5)
# b.set_title(f'tau vs H$_0$ {nombre_C1}\n300 kHz')

# b.set_xlabel('H$_0$ (kA/m)')
# b.set_ylabel('tau (ns)')
# b.legend(ncol=2, loc='best')
# b.grid()

# b.set_xticks(ticks)  
# b.set_xticklabels([str(h) for h in ticks])  
# plt.show()

# #%% Hc 
# fig,c = plt.subplots(nrows=1, figsize=(7,5), constrained_layout=True)
# c.errorbar(x=H0,y=[h.n for h in Hc_C1], yerr=[h.s for h in Hc_C1], fmt='.',ls='-', label='C1', capsize=5)
# c.set_title('Hc vs H$_0$')

# c.set_xlabel('H$_0$ (kA/m)')        
# c.set_ylabel('H$_c$ (kA/m)')
# c.legend(ncol=2, loc='best')
# c.grid()

# c.set_xticks(H)  
# c.set_xticklabels(H)  
# plt.show()


#%%

fig00.savefig('00_ciclos_promedio_C1.png',dpi=300)
fig01.savefig('01_templogs_C1.png',dpi=300)
fig02.savefig('02_comparativa_ESAR_tau_Hc_vs_H0_C1.png',dpi=300)

# %%
