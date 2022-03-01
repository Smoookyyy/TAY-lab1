import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



def dead_zone_scalar(x, width = 0.5):   
    if np.abs(x)<width:
        return 0
    elif x>0:
        return x-width
    else:
        return x+width
        
dead_zone = np.vectorize(dead_zone_scalar, otypes=[np.float], excluded=['width'])

def saturation_scalar(x, width = 0.5):
   if np.abs(x)<width:
        return x
   elif x>width:
        return (width)
   else:
        return (-width)

saturation = np.vectorize(saturation_scalar, otypes=[np.float], excluded=['width'])


variant = 5 

test_signal_duration = 100
test_sig_ampl = 1 + variant * 0.1
test_sig_freq = 1 + variant * 3.5
non_lin_param_1 = 0.5 + variant * 0.1
lin_param_k = 0.5 + variant * 0.3
lin_param_T = 0.1 + variant * 0.2

print("Вариант номер {}".format(variant))
print("Амплитуда тестового сигнала: {:.2}".format(test_sig_ampl))
print("Частота тестового сигнала: {:.2} Гц".format(test_sig_freq))
print("Длительность тестового сигнала: {} с".format(test_signal_duration))
print("Параметр нелинейностей 1: {:.2}".format(non_lin_param_1))

dt = 0.001
print("Период дискретизации сигнала: {:.2} с".format(dt))

t = np.arange(0, test_signal_duration, dt)

print("Размерность массива время: {}".format(t.shape))
print("Содержимое массива: {}".format(t))

sig_sin = test_sig_ampl * np.sin(test_sig_freq * t * 2 * np.pi)
sig_meandr = test_sig_ampl * signal.square(test_sig_freq * t * 2 * np.pi)
sig_saw = test_sig_ampl * signal.sawtooth(test_sig_freq * t * 2 * np.pi)

relay_sin = np.sign(sig_sin) 
relay_meandr = np.sign(sig_meandr)
relay_saw = np.sign(sig_saw)   #Идеальное реле

print("Размерность сигнала синусоида: {}".format(sig_sin.shape[0]))
print("Размерность сигнала: {}".format(sig_sin.shape))
print("Содержимое массива сигнала: {}".format(sig_sin))

plt.figure('Синусоида')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4.5, 4.5)
plt.xlim(0, 1)
plt.plot(t, sig_sin)
plt.show()

plt.figure('Меандр')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4.5, 4.5)
plt.xlim(0, 1)
plt.plot(t, sig_meandr)
plt.show()

plt.figure('Пила')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4.5, 4.5)
plt.xlim(0, 1)
plt.plot(t, sig_saw)
plt.show()

sig_sin_spec = np.abs(np.fft.fft(sig_sin))
sig_meandr_spec = np.abs(np.fft.fft(sig_meandr))
sig_saw_spec = np.abs(np.fft.fft(sig_saw))
print("Размерность массива спектра синусоиды: {}".format(sig_sin_spec.shape))
print("Содержимое массива спектра: {}".format(sig_sin_spec))

####################################################33

plt.figure('Спектр Синусоиды по каналам')
freqs = np.fft.fftfreq(sig_sin.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_sin_spec)
plt.show()

plt.figure('Спектр Меандра по каналам')
freqs = np.fft.fftfreq(sig_meandr.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_meandr_spec)
plt.show()

plt.figure('Спектр Пилы по каналам')
freqs = np.fft.fftfreq(sig_saw.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_saw_spec)
plt.show()

#####################################################333

plt.figure('Идеальное реле на синусоиду')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_sin, t, relay_sin)
plt.show()


plt.figure('Мертвая зона на синусоиду')
sig_sin_dz = dead_zone(sig_sin, non_lin_param_1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_sin, t, sig_sin_dz)
plt.show()



plt.figure('Насыщение на синусоиду')
sat_sin_dz = saturation(sig_sin, non_lin_param_1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_sin, t, sat_sin_dz)
plt.show()

###################################################


plt.figure('Идеальное реле на меандр')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_meandr, t, relay_meandr)
plt.show()


plt.figure('Мертвая зона на меандр')
sig_meandr_dz = dead_zone(sig_meandr, non_lin_param_1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_meandr, t, sig_meandr_dz)
plt.show()



plt.figure('Насыщение на меандр')
sat_meandr_dz = saturation(sig_meandr, non_lin_param_1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_meandr, t, sat_meandr_dz)
plt.show()

############################################

plt.figure('Идеальное реле на Пилу')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_saw, t, relay_saw)
plt.show()


plt.figure('Мертвая зона на Пилу')
sig_saw_dz = dead_zone(sig_saw, non_lin_param_1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_saw, t, sig_saw_dz)
plt.show()



plt.figure('Насыщение на Пилу')
sat_saw_dz = saturation(sig_saw, non_lin_param_1)
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
plt.ylim(-4, 4)
plt.xlim(0, 0.5)
plt.plot(t, sig_saw, t, sat_saw_dz)
plt.show()

###########################################

sig_sin_spec_relay = np.abs(np.fft.fft(relay_sin))
sig_meandr_spec_relay = np.abs(np.fft.fft(relay_meandr))
sig_saw_spec_relay = np.abs(np.fft.fft(relay_saw))

plt.figure('Спектр Синусоиды по каналам с Идеальным реле')
freqs = np.fft.fftfreq(relay_sin.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_sin_spec_relay)
plt.show()

plt.figure('Спектр Меандра по каналам с Идеальным реле')
freqs = np.fft.fftfreq(relay_meandr.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_meandr_spec_relay)
plt.show()

plt.figure('Спектр Пилы по каналам с Идеальным реле')
freqs = np.fft.fftfreq(relay_saw.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_saw_spec_relay)
plt.show()      

###############

sig_sin_spec_dz = np.abs(np.fft.fft(sig_sin_dz))
sig_meandr_spec_dz = np.abs(np.fft.fft(sig_meandr_dz))
sig_saw_spec_dz = np.abs(np.fft.fft(sig_saw_dz))

plt.figure('Спектр Синусоиды по каналам Мертвая зона')
freqs = np.fft.fftfreq(sig_sin_dz.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_sin_spec_dz)
plt.show()

plt.figure('Спектр Меандра по каналам Мертвая зона')
freqs = np.fft.fftfreq(sig_meandr_dz.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_meandr_spec_dz)
plt.show()

plt.figure('Спектр Пилы по каналам Мертвая зона')
freqs = np.fft.fftfreq(sig_saw_dz.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_saw_spec_dz)
plt.show()

###################

sat_sin_spec_dz = np.abs(np.fft.fft(sat_sin_dz))
sat_meandr_spec_dz = np.abs(np.fft.fft(sat_meandr_dz))
sat_saw_spec_dz = np.abs(np.fft.fft(sat_saw_dz))

plt.figure('Спектр Синусоиды по каналам Насыщение')
freqs = np.fft.fftfreq(sat_sin_dz.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sat_sin_spec_dz)
plt.show()

plt.figure('Спектр Меандра по каналам Насыщение')
freqs = np.fft.fftfreq(sat_meandr_dz.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sat_meandr_spec_dz)
plt.show()

plt.figure('Спектр Пилы по каналам Насыщение')
freqs = np.fft.fftfreq(sat_saw_dz.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sat_saw_spec_dz)
plt.show()

#################### Фильтр с реле

k = lin_param_k
T = lin_param_T
B = [ k/(1+T/dt) ]
A = [1, -1/(1+dt/T)]
sig_sin_relay_lb = signal.lfilter(B, A, relay_sin) 
sig_meandr_relay_lb = signal.lfilter(B, A, relay_meandr)
sig_saw_relay_lb = signal.lfilter(B, A, relay_saw)


plt.figure('Синусоида после Идеальное реле с фильтром')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, relay_sin, t, sig_sin_relay_lb)
plt.show()

plt.figure('Меандр после Идеальное реле с фильтром')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_meandr, t, relay_meandr, t, sig_meandr_relay_lb)
plt.show()

plt.figure('Пила после Идеальное реле с фильтром') 
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, relay_saw, t, sig_saw_relay_lb)
plt.show()

#################### Фильтр с мертвой зоной
sig_sin_dz_lb = signal.lfilter(B, A, sig_sin_dz) 
sig_meandr_dz_lb = signal.lfilter(B, A, sig_meandr_dz)
sig_saw_dz_lb = signal.lfilter(B, A, sig_saw_dz)

plt.figure('Синусоида после мертвой зоны с фильтром')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sig_sin_dz, t, sig_sin_dz_lb)
plt.show()

plt.figure('Меандр после мертвой зоны с фильтром')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_meandr, t, sig_meandr_dz, t, sig_meandr_dz_lb)
plt.show()

plt.figure('Пила после мертвой зоны с фильтром') 
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sig_saw_dz, t, sig_saw_dz_lb)
plt.show()

#################### Фильтр с насыщением
sat_sin_dz_lb = signal.lfilter(B, A, sat_sin_dz) 
sat_meandr_dz_lb = signal.lfilter(B, A, sat_meandr_dz)
sat_saw_dz_lb = signal.lfilter(B, A, sat_saw_dz)

plt.figure('Синусоида после насыщения с фильтром')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_sin, t, sat_sin_dz, t, sat_sin_dz_lb)
plt.show()

plt.figure('Меандр после насыщения с фильтром')
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_meandr, t, sat_meandr_dz, t, sat_meandr_dz_lb)
plt.show()

plt.figure('Пила после насыщения с фильтром') 
plt.grid()
plt.xlabel('Time, s')
plt.ylabel('Ampl')
#plt.ylim(-0.1, 0.1)
plt.xlim(0, 1)
plt.plot(t, sig_saw, t, sat_saw_dz, t, sat_saw_dz_lb)
plt.show()

#################### Спектр фильтра с идеальным реле

sig_sin_spec_relay_lb = np.abs(np.fft.fft(sig_sin_relay_lb))
sig_meandr_spec_relay_lb = np.abs(np.fft.fft(sig_meandr_relay_lb))
sig_saw_spec_relay_lb = np.abs(np.fft.fft(sig_saw_relay_lb))

plt.figure('Спектр Синусоиды с Идеальным реле фильтр')
freqs = np.fft.fftfreq(sig_sin_relay_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_sin_spec_relay_lb)
plt.show()

plt.figure('Спектр Меандра по каналам с Идеальным реле фильтр')
freqs = np.fft.fftfreq(sig_meandr_relay_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_meandr_spec_relay_lb)
plt.show()

plt.figure('Спектр Пилы по каналам с Идеальным реле фильтр')
freqs = np.fft.fftfreq(sig_saw_relay_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_saw_spec_relay_lb)
plt.show()

#################### Спектр фильтра с мертвой зоны
sig_sin_spec_dz_lb = np.abs(np.fft.fft(sig_sin_dz_lb))
sig_meandr_spec_dz_lb = np.abs(np.fft.fft(sig_meandr_dz_lb))
sig_saw_spec_dz_lb = np.abs(np.fft.fft(sig_saw_dz_lb))

plt.figure('Спектр Синусоиды по каналам Мертвая зона фильтр')
freqs = np.fft.fftfreq(sig_sin_dz_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_sin_spec_dz_lb)
plt.show()

plt.figure('Спектр Меандра по каналам Мертвая зона фильтр')
freqs = np.fft.fftfreq(sig_meandr_dz_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_meandr_spec_dz_lb)
plt.show()

plt.figure('Спектр Пилы по каналам Мертвая зона фильтр')
freqs = np.fft.fftfreq(sig_saw_dz_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sig_saw_spec_dz_lb)
plt.show()

#################### Спектр фильтра с насыщением

sat_sin_spec_dz_lb = np.abs(np.fft.fft(sat_sin_dz_lb))
sat_meandr_spec_dz_lb = np.abs(np.fft.fft(sat_meandr_dz_lb))
sat_saw_spec_dz_lb = np.abs(np.fft.fft(sat_saw_dz_lb))

plt.figure('Спектр Синусоиды по каналам с Насыщением фильтр')
freqs = np.fft.fftfreq(sat_sin_dz_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sat_sin_spec_dz_lb)
plt.show()

plt.figure('Спектр Меандра по каналам с Насыщением фильтр')
freqs = np.fft.fftfreq(sat_meandr_dz_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sat_meandr_spec_dz_lb)
plt.show()

plt.figure('Спектр Пилы по каналам с Насыщением фильтр')
freqs = np.fft.fftfreq(sat_saw_dz_lb.shape[0], dt)
plt.grid()
plt.xlabel('Частота, Гц')
plt.xlim(-25, 25)
plt.plot(freqs, sat_saw_spec_dz_lb)
plt.show()