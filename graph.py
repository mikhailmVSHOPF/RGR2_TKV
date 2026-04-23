import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.gridspec import GridSpec

# Настройка параметров по умолчанию
DEFAULT_PARAMS = {
    'N': 40,
    'k': 1.0,
    'm': 1.0,
    'M': 10.0,
    'fb': 2.0,
    'fa': 10.0,
    'jamma': 0.0001,
    'w0_min': 0.001,
    'w0_max': 2.0,
    'n_points': 1000
}

def get_amplitudes(k_idx, w0, params):
    """Вычисление амплитуд для заданной моды"""
    ksi = np.pi * k_idx / params['N']
    
    phi = ((w0/params['omega'])**2 - 1 + np.exp(-1j * ksi)) / \
          ((w0/params['omega'])**2 - 1 + np.exp(1j * ksi))
    
    n = params['N']/2
    
    eq1 = (2*params['k'] - params['M']*w0**2 + 2j*params['jamma']*w0) * \
          (np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi*n))
    
    eq2 = params['k'] * (np.exp(1j*ksi*(n+1)) - phi*np.exp(1j*ksi)*np.exp(-1j*ksi*(n+1)) + \
                        np.exp(1j*ksi*n) - phi*np.exp(1j*ksi)*np.exp(-1j*ksi*n))
    
    eq3 = (2*params['k'] - params['m']*w0**2 + 2j*params['jamma']*w0) * \
          (np.exp(1j*ksi*n) - phi*np.exp(1j*ksi)*np.exp(-1j*ksi*n))
    
    eq4 = params['k'] * (np.exp(1j*ksi*n) - phi*np.exp(-1j*ksi*n) + \
                        np.exp(1j*ksi*(n-1)) - phi*np.exp(-1j*ksi*(n-1)))
    
    B_val = (params['fb']/2j + (eq2/eq3)*params['fa']/2j) / (eq1 - (eq4/eq3)*eq2)
    A_val = (B_val * eq1 - params['fb']/2j) / eq2
    
    return B_val, A_val

def compute_amplitude(w0_array, params, mode_range=None):
    """Вычисление суммарной амплитуды по всем модам"""
    params['omega'] = np.sqrt(params['k']/params['m'])
    
    if mode_range is None:
        mode_range = range(1, params['N'])
    
    sum_B = np.zeros(len(w0_array), dtype=complex)
    sum_A = np.zeros(len(w0_array), dtype=complex)
    
    for ind in mode_range:
        B_val, A_val = get_amplitudes(ind, w0_array, params)
        sum_B += B_val
        sum_A += A_val
    
    return np.abs(sum_B), np.abs(sum_A)

class ResonanceInterface:
    """Интерактивный интерфейс для анализа резонансов"""
    
    def __init__(self):
        # Инициализация параметров
        self.params = DEFAULT_PARAMS.copy()
        self.params['omega'] = np.sqrt(self.params['k']/self.params['m'])
        
        # Частотная сетка
        self.w0 = np.linspace(self.params['w0_min'], 
                              self.params['w0_max'], 
                              self.params['n_points'])
        
        # Вычисление начальных амплитуд
        self.amplitude_B, self.amplitude_A = compute_amplitude(self.w0, self.params)
        
        # Создание фигуры
        self.fig = plt.figure(figsize=(16, 10))
        self.setup_layout()
        self.setup_controls()
        self.update_plots()
        
        plt.show()
    
    def setup_layout(self):
        """Настройка layout графиков"""
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[1, 1, 2], hspace=0.3, wspace=0.3)
        
        # Основные графики
        self.ax_main = self.fig.add_subplot(gs[0, :])
        self.ax_log = self.fig.add_subplot(gs[1, :])
        self.ax_mode_contrib = self.fig.add_subplot(gs[2, 0])
        self.ax_params = self.fig.add_subplot(gs[2, 1])
        
        # Настройка осей
        self.ax_main.set_xlabel(r'Частота $\omega_0$')
        self.ax_main.set_ylabel('Амплитуда B')
        self.ax_main.set_title('Амплитудный спектр (линейный масштаб)')
        self.ax_main.grid(True, alpha=0.3)
        
        self.ax_log.set_xlabel(r'Частота $\omega_0$')
        self.ax_log.set_ylabel('Амплитуда B (лог)')
        self.ax_log.set_title('Амплитудный спектр (логарифмический масштаб)')
        self.ax_log.grid(True, alpha=0.3)
        self.ax_log.set_yscale('log')
        
        self.ax_mode_contrib.set_title('Вклад мод в амплитуду')
        self.ax_mode_contrib.set_xlabel('Номер моды')
        self.ax_mode_contrib.set_ylabel('Относительный вклад')
        self.ax_mode_contrib.grid(True, alpha=0.3)
        
        self.ax_params.axis('off')
        self.ax_params.set_title('Параметры системы')
    
    def setup_controls(self):
        """Создание ползунков управления"""
        # Позиции для ползунков
        slider_positions = {
            'N': [0.1, 0.02, 0.3, 0.03],
            'k': [0.1, 0.07, 0.3, 0.03],
            'm': [0.1, 0.12, 0.3, 0.03],
            'M': [0.1, 0.17, 0.3, 0.03],
            'fb': [0.1, 0.22, 0.3, 0.03],
            'fa': [0.1, 0.27, 0.3, 0.03],
            'jamma': [0.1, 0.32, 0.3, 0.03],
            'w0_max': [0.6, 0.02, 0.3, 0.03]
        }
        
        # Создание ползунков
        self.sliders = {}
        
        self.sliders['N'] = Slider(
            plt.axes(slider_positions['N']), 'N (узлы)', 
            10, 100, valinit=self.params['N'], valstep=1
        )
        
        self.sliders['k'] = Slider(
            plt.axes(slider_positions['k']), 'k (жесткость)', 
            0.1, 5.0, valinit=self.params['k']
        )
        
        self.sliders['m'] = Slider(
            plt.axes(slider_positions['m']), 'm (масса легкая)', 
            0.1, 5.0, valinit=self.params['m']
        )
        
        self.sliders['M'] = Slider(
            plt.axes(slider_positions['M']), 'M (масса тяжелая)', 
            1.0, 50.0, valinit=self.params['M']
        )
        
        self.sliders['fb'] = Slider(
            plt.axes(slider_positions['fb']), 'fb (амплитуда B)', 
            0.1, 20.0, valinit=self.params['fb']
        )
        
        self.sliders['fa'] = Slider(
            plt.axes(slider_positions['fa']), 'fa (амплитуда A)', 
            0.1, 20.0, valinit=self.params['fa']
        )
        
        self.sliders['jamma'] = Slider(
            plt.axes(slider_positions['jamma']), 'jamma (затухание)', 
            1e-6, 0.01, valinit=self.params['jamma'], valformat='%.1e'
        )
        
        self.sliders['w0_max'] = Slider(
            plt.axes(slider_positions['w0_max']), 'Макс. частота', 
            1.0, 5.0, valinit=self.params['w0_max']
        )
        
        # Привязка обработчиков
        for slider in self.sliders.values():
            slider.on_changed(self.update_params)
        
        # Кнопка сброса
        from matplotlib.widgets import Button
        reset_ax = plt.axes([0.6, 0.32, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Сброс')
        self.reset_button.on_clicked(self.reset_params)
        
        # Переключатель типа амплитуды
        radio_ax = plt.axes([0.75, 0.25, 0.15, 0.1])
        self.radio = RadioButtons(radio_ax, ['Амплитуда B', 'Амплитуда A'])
        self.radio.on_clicked(self.toggle_amplitude)
        
        self.show_B = True
    
    def update_params(self, val):
        """Обновление параметров при изменении ползунков"""
        # Обновление значений
        self.params['N'] = int(self.sliders['N'].val)
        self.params['k'] = self.sliders['k'].val
        self.params['m'] = self.sliders['m'].val
        self.params['M'] = self.sliders['M'].val
        self.params['fb'] = self.sliders['fb'].val
        self.params['fa'] = self.sliders['fa'].val
        self.params['jamma'] = self.sliders['jamma'].val
        self.params['w0_max'] = self.sliders['w0_max'].val
        
        # Пересчет параметров
        self.params['omega'] = np.sqrt(self.params['k']/self.params['m'])
        
        # Обновление частотной сетки
        self.w0 = np.linspace(self.params['w0_min'], 
                              self.params['w0_max'], 
                              self.params['n_points'])
        
        # Пересчет амплитуд
        self.amplitude_B, self.amplitude_A = compute_amplitude(self.w0, self.params)
        
        # Обновление графиков
        self.update_plots()
    
    def reset_params(self, event):
        """Сброс всех параметров к значениям по умолчанию"""
        for key, slider in self.sliders.items():
            slider.set_val(DEFAULT_PARAMS[key])
    
    def toggle_amplitude(self, label):
        """Переключение между амплитудами B и A"""
        self.show_B = (label == 'Амплитуда B')
        self.update_plots()
    
    def update_plots(self):
        """Обновление всех графиков"""
        # Выбор текущей амплитуды
        current_amp = self.amplitude_B if self.show_B else self.amplitude_A
        ylabel = 'Амплитуда B' if self.show_B else 'Амплитуда A'
        
        # Обновление основного графика
        self.ax_main.clear()
        self.ax_main.plot(self.w0, current_amp, 'b-', linewidth=1.5)
        self.ax_main.set_xlabel(r'Частота $\omega_0$')
        self.ax_main.set_ylabel(ylabel)
        self.ax_main.set_title(f'{ylabel} (линейный масштаб)')
        self.ax_main.grid(True, alpha=0.3)
        
        # Автоматическая настройка пределов по Y (обрезаем выбросы)
        percentile_95 = np.percentile(current_amp, 95)
        self.ax_main.set_ylim([0, percentile_95])
        
        # Обновление логарифмического графика
        self.ax_log.clear()
        self.ax_log.plot(self.w0, current_amp, 'r-', linewidth=1.5)
        self.ax_log.set_xlabel(r'Частота $\omega_0$')
        self.ax_log.set_ylabel(ylabel + ' (лог)')
        self.ax_log.set_title(f'{ylabel} (логарифмический масштаб)')
        self.ax_log.set_yscale('log')
        self.ax_log.grid(True, alpha=0.3)
        
        # Расчет вклада мод при резонансной частоте
        if len(current_amp) > 0:
            peak_idx = np.argmax(current_amp)
            peak_freq = self.w0[peak_idx]
            
            mode_contributions = []
            for ind in range(1, self.params['N']):
                B, A = get_amplitudes(ind, np.array([peak_freq]), self.params)
                mode_contributions.append(np.abs(B if self.show_B else A)[0])
            
            mode_contributions = np.array(mode_contributions)
            if np.sum(mode_contributions) > 0:
                mode_contributions /= np.sum(mode_contributions)
            
            # Обновление графика вклада мод
            self.ax_mode_contrib.clear()
            modes = range(1, self.params['N'])
            self.ax_mode_contrib.bar(modes, mode_contributions, alpha=0.7)
            self.ax_mode_contrib.set_xlabel('Номер моды')
            self.ax_mode_contrib.set_ylabel('Относительный вклад')
            self.ax_mode_contrib.set_title(f'Вклад мод при ω₀ = {peak_freq:.3f}')
            self.ax_mode_contrib.grid(True, alpha=0.3)
        
        # Отображение текущих параметров
        self.ax_params.clear()
        self.ax_params.axis('off')
        param_text = f"""Текущие параметры:
        
N = {self.params['N']}
k = {self.params['k']:.3f}
m = {self.params['m']:.3f}
M = {self.params['M']:.3f}
fb = {self.params['fb']:.3f}
fa = {self.params['fa']:.3f}
γ = {self.params['jamma']:.2e}
ω_max = {self.params['w0_max']:.3f}
ω₀ резонанса = {self.w0[np.argmax(current_amp)]:.3f}"""
        
        self.ax_params.text(0.1, 0.9, param_text, transform=self.ax_params.transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Перерисовка
        self.fig.canvas.draw_idle()

# Запуск интерфейса
if __name__ == "__main__":
    interface = ResonanceInterface()