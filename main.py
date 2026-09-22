#
# 
#
# Copyright (c) 2026 Reggi Aryunadi
# 
# This software is created for Remote Calibration services at
# the Time and Frequency Laboratory of SNSU-BSN (National Metrology Institute of Indonesia)
#
# Updated 13-05-2026 18:30 UTC

import os
import sys

# 1. BYPASS CONSOLE (Mencegah error saat Matplotlib mencoba print log di background)
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# 2. FUNGSI RESOURCE PATH (Seperti yang kita bahas sebelumnya)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 3. SETTING BACKEND MATPLOTLIB UNTUK PYSIDE6
import matplotlib
# Memaksa matplotlib menggunakan engine Qt yang kompatibel dengan PySide6
matplotlib.use('qtagg') 
import matplotlib.pyplot as plt

import math
import requests  
import pandas as pd
import numpy as np  
from pathlib import Path
from datetime import datetime, timezone, timedelta
from astropy.time import Time
from scipy.stats import linregress

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QTextEdit, QFileDialog, QLineEdit, QPushButton, QTabWidget,
    QGroupBox, QDateTimeEdit, QFrame, QTableWidget, QTableWidgetItem, 
    QHeaderView, QComboBox, QMessageBox
)
from PySide6.QtGui import QFont, QPixmap, QIcon, QDesktopServices
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QDateTime, QUrl

class JumpDetectionThread(QThread):
    activity_update = Signal(str)
    report_update = Signal(str)
    error_update = Signal(str)
    finished_update = Signal(str)
    graph_data_update = Signal(object)

    def __init__(self, folder_dir):
        super().__init__()
        self.folder_dir = folder_dir
        self.df_result = pd.DataFrame()
        self.jumps_result = []

    def parse_cggtts(self, filepath):
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            header_idx = -1
            header_cols = []
            for i, line in enumerate(lines):
                if 'MJD' in line and 'STTIME' in line:
                    header_idx = i
                    header_cols = line.strip().split()
                    break
            
            if header_idx == -1: return pd.DataFrame()

            ref_col_name = 'REFSYS' if 'REFSYS' in header_cols else 'REFGPS'
            sat_col_name = 'SAT' if 'SAT' in header_cols else 'PRN'
            
            idx_sat = header_cols.index(sat_col_name)
            idx_mjd = header_cols.index('MJD')
            idx_time = header_cols.index('STTIME')
            idx_ref = header_cols.index(ref_col_name)

            for line in lines[header_idx+2:]:
                parts = line.strip().split()
                if len(parts) > max(idx_sat, idx_mjd, idx_time, idx_ref):
                    sat = parts[idx_sat]
                    if not sat[0].isdigit(): sat = sat[1:] 
                    
                    data.append({
                        'SAT/PRN': int(sat),
                        'MJD': int(parts[idx_mjd]),
                        'STTIME': int(parts[idx_time]),
                        'REF': float(parts[idx_ref]) / 10.0,
                    })
        except: pass
        return pd.DataFrame(data)

    def run(self):
        try:
            self.activity_update.emit("Tahap 1: Membaca semua file di folder...")
            files = [f for f in Path(self.folder_dir).glob('*') if f.is_file()]
            if not files:
                self.error_update.emit("Folder kosong atau tidak ditemukan.")
                return

            raw_dfs = []
            for f in files:
                df_file = self.parse_cggtts(f)
                if not df_file.empty:
                    raw_dfs.append(df_file)

            if not raw_dfs:
                self.error_update.emit("Tidak ditemukan data CGGTTS valid di dalam folder.")
                return

            df_all = pd.concat(raw_dfs, ignore_index=True)

            self.activity_update.emit("Tahap 2 & 3: Mengelompokkan dan merata-rata per waktu...")
            df_grouped = df_all.groupby(['MJD', 'STTIME'], as_index=False)['REF'].mean()
            df_grouped = df_grouped.sort_values(['MJD', 'STTIME']).reset_index(drop=True)

            if len(df_grouped) < 2:
                self.error_update.emit("Data kurang dari 2 titik waktu untuk analisis loncatan dan Allan Variance.")
                return

            min_mjd = df_grouped['MJD'].min()
            max_mjd = df_grouped['MJD'].max()
            epoch = datetime(1858, 11, 17)
            start_date = epoch + timedelta(days=float(min_mjd))
            end_date = epoch + timedelta(days=float(max_mjd))

            y_vals = df_grouped['REF'].values
            if len(y_vals) > 2:
                allan_var = np.mean((y_vals[2:] - 2*y_vals[1:-1] + y_vals[:-2])**2) / 2
            else:
                allan_var = 0.0

            self.activity_update.emit("Tahap 4: Mendeteksi loncatan (jump >= 100 ns)...")
            jumps = []
            for i in range(1, len(df_grouped)):
                mjd_curr = df_grouped.loc[i, 'MJD']
                sttime_curr = df_grouped.loc[i, 'STTIME']
                diff = y_vals[i] - y_vals[i-1]
                if abs(diff) >= 100.0:
                    jumps.append({
                        'MJD': mjd_curr,
                        'STTIME': sttime_curr,
                        'Delta (ns)': diff
                    })

            self.activity_update.emit("Tahap 5: Menyusun laporan analisis...")
            report = "=== LAPORAN JUMP DETECTION ===\n\n"
            report += f"Jumlah Data Mentah Teranalisis : {len(df_all)} baris\n"
            report += f"Jumlah Titik Waktu Rata-rata   : {len(df_grouped)} titik\n"
            report += f"Rentang MJD                    : {min_mjd} s.d. {max_mjd}\n"
            report += f"Rentang Tanggal                : {start_date.strftime('%Y-%m-%d')} s.d. {end_date.strftime('%Y-%m-%d')}\n"
            report += f"Allan Variance (Semua Data)    : {allan_var:.6e} ns²\n\n"

            if jumps:
                report += f"STATUS: Ditemukan {len(jumps)} loncatan (Jump >= 100 ns):\n"
                report += "-"*50 + "\n"
                report += f"{'No':<4} | {'MJD':<8} | {'STTIME':<8} | {'Besar Loncatan (ns)'}\n"
                report += "-"*50 + "\n"
                for idx, j in enumerate(jumps, 1):
                    m_val = j['MJD']
                    s_val = j['STTIME']
                    d_val = j['Delta (ns)']
                    report += f"{idx:<4} | {m_val:<8} | {s_val:<8} | {d_val:.3f}\n"
            else:
                report += "STATUS: Tidak ada loncatan (jump) >= 100 ns yang terdeteksi.\n"

            self.df_result = df_grouped
            self.jumps_result = jumps

            self.graph_data_update.emit(df_grouped)
            self.report_update.emit(report)
            self.finished_update.emit("Selesai")

        except Exception as e:
            self.error_update.emit(str(e))

# ==========================================
# THREAD ANALISIS MONTE CARLO
# ==========================================
class MonteCarloThread(QThread):
    activity_update = Signal(str)
    report_update = Signal(str)
    graph_data_update = Signal(object)
    finished_update = Signal(str)
    error_update = Signal(str)

    def __init__(self, inputs_data, N_samples):
        super().__init__()
        self.inputs_data = inputs_data
        self.N_samples = N_samples

    def run(self):
        try:
            self.activity_update.emit("Memulai simulasi Monte Carlo...")
            np.random.seed(42)
            N = self.N_samples
            samples = np.zeros((N, len(self.inputs_data)))
            
            for idx, (name, dist, u) in enumerate(self.inputs_data):
                if u == 0:
                    arr = np.zeros(N)
                else:
                    if dist.lower() == "normal":
                        arr = np.random.normal(0.0, u, N)
                    elif dist.lower() in ["rect", "rectangular"]:
                        a = math.sqrt(3) * u
                        arr = np.random.uniform(-a, a, N)
                    else:
                        raise ValueError(f"Distribusi {dist} tidak dikenali")
                samples[:, idx] = arr

            self.activity_update.emit("Menghitung gabungan sampel...")
            Y = np.sum(samples, axis=1)
            y_mean = np.mean(Y)
            u_c = np.std(Y, ddof=1)
            
            alpha = (100 - 95) / 2
            low = np.percentile(Y, alpha)
            high = np.percentile(Y, 100 - alpha)
            halfwidth = (high - low) / 2
            k = halfwidth / u_c if u_c != 0 else 0

            report = "=== HASIL EVALUASI MONTE CARLO (95%) ===\n\n"
            report += f"Jumlah Sampel (N) : {N}\n\n"
            report += f"Y Mean            : {y_mean:.6e}\n"
            report += f"u_c (Std Dev)     : {u_c:.6e}\n"
            report += f"Interval (95%)    : [{low:.6e},  {high:.6e}]\n"
            report += f"Expanded Uncert.  : {halfwidth:.6e}\n"
            report += f"Coverage Factor k : {k:.6f}\n"

            self.graph_data_update.emit(Y)
            self.report_update.emit(report)
            self.finished_update.emit("Selesai")

        except Exception as e:
            self.error_update.emit(str(e))

# ==========================================
# THREAD ANALISIS UTC(K)
# ==========================================
class UTCAnalysisThread(QThread):
    activity_update = Signal(str)
    report_update = Signal(str)
    error_update = Signal(str)
    finished_update = Signal(str)
    graph_data_update = Signal(object, object, object, object)

    def __init__(self, start_mjd, stop_mjd, utck, save_path):
        super().__init__()
        self.start_mjd = start_mjd
        self.stop_mjd = stop_mjd
        self.utck = utck
        self.save_path = save_path

    def run(self):
        try:
            self.activity_update.emit(f"Mengambil data API BIPM untuk UTC({self.utck})...")
            url = f"https://webtai.bipm.org/api/v0.2-beta/get-data.html?scale=utc&lab={self.utck}&outfile=txt&mjd1={self.start_mjd}&mjd2={self.stop_mjd}"
            r = requests.get(url, timeout=15)
            
            if r.status_code != 200:
                self.error_update.emit(f"Gagal narik data. HTTP Status: {r.status_code}")
                return

            data = r.text.split('\n')
            timestamps = []
            values = []

            for line in data:
                line = line.strip()
                if line and "UTC" not in line and "ns" not in line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        timestamps.append(float(parts[0]))
                        values.append(float(parts[1]))

            if not timestamps:
                self.error_update.emit("Data tidak ditemukan untuk rentang MJD tersebut.")
                return

            self.activity_update.emit("Menghitung regresi linier...")
            x = np.array(timestamps)
            y = np.array(values)
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            regression_values = slope * x + intercept
            
            mean_y = np.mean(y)
            total_sum_of_squares = np.sum((y - mean_y) ** 2)
            if total_sum_of_squares == 0:
                r_squared = 0
            else:
                residual_sum_of_squares = np.sum((y - regression_values) ** 2)
                r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

            report = f"=== HASIL ANALISIS UTC({self.utck}) ===\n"
            report += f"MJD Awal  : {self.start_mjd}\n"
            report += f"MJD Akhir : {self.stop_mjd}\n"
            report += f"Total Data: {len(x)} titik\n\n"
            report += f"Persamaan Regresi Linier:\n"
            report += f"Y = {slope:.5f}X + ({intercept:.5f})\n\n"
            report += f"Slope     : {slope:.5f} ns/day\n"
            report += f"Intercept : {intercept:.5f} ns\n"
            report += f"R-squared : {r_squared:.5f}\n"

            if self.save_path:
                self.activity_update.emit("Menyimpan ke Excel...")
                df = pd.DataFrame({
                    'MJD': x,
                    f'UTC-UTC({self.utck}) (ns)': y,
                    'Regresi (ns)': regression_values
                })
                with pd.ExcelWriter(self.save_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name="Data_API", index=False)
                    df_stats = pd.DataFrame({
                        'Parameter': ['Slope', 'Intercept', 'R-squared', 'Formula Linier'],
                        'Value': [slope, intercept, r_squared, f"Y = {slope:.5f}X + {intercept:.5f}"]
                    })
                    df_stats.to_excel(writer, sheet_name="Statistik", index=False)
                report += f"\nData berhasil disimpan di:\n{self.save_path}"

            self.graph_data_update.emit(x, y, regression_values, f"Y = {slope:.5f}X + {intercept:.5f}")
            self.report_update.emit(report)
            self.finished_update.emit("Selesai")

        except Exception as e:
            self.error_update.emit(str(e))

# ==========================================
# THREAD CEK KELENGKAPAN DATA
# ==========================================
class CheckDataThread(QThread):
    activity_update = Signal(str)      
    progress_update = Signal(int)      
    report_update = Signal(str)      
    error_update = Signal(str)
    finished_update = Signal(str)
    graph_data_update = Signal(object, object) # Mengirim data grafik (Std, UUT)

    def __init__(self, dir_std, dir_uut):
        super().__init__()
        self.dir_std = dir_std
        self.dir_uut = dir_uut
        self.df_std_grouped = pd.DataFrame()
        self.df_uut_grouped = pd.DataFrame()

    def parse_cggtts_full(self, filepath):
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            header_idx = -1
            header_cols = []
            for i, line in enumerate(lines):
                if 'MJD' in line and 'STTIME' in line:
                    header_idx = i
                    header_cols = line.strip().split()
                    break
            
            if header_idx == -1: return pd.DataFrame()

            ref_col_name = 'REFSYS' if 'REFSYS' in header_cols else 'REFGPS'
            sat_col_name = 'SAT' if 'SAT' in header_cols else 'PRN'
            
            idx_sat = header_cols.index(sat_col_name)
            idx_mjd = header_cols.index('MJD')
            idx_time = header_cols.index('STTIME')
            idx_ref = header_cols.index(ref_col_name)

            for line in lines[header_idx+2:]:
                parts = line.strip().split()
                if len(parts) > max(idx_sat, idx_mjd, idx_time, idx_ref):
                    sat = parts[idx_sat]
                    if not sat[0].isdigit(): sat = sat[1:] 
                    
                    data.append({
                        'SAT/PRN': int(sat),
                        'MJD': int(parts[idx_mjd]),
                        'STTIME': int(parts[idx_time]),
                        'REF': float(parts[idx_ref]) / 10.0,
                    })
        except: pass
        return pd.DataFrame(data)

    def analyze_folder_with_jump(self, folder_path, title, start_prog, end_prog):
        files = [f for f in Path(folder_path).glob('*') if f.is_file()]
        if not files:
            return f"=== DATA {title.upper()} ===\nFolder kosong atau tidak ditemukan.\n\n", pd.DataFrame()
        
        raw_dfs = []
        total_files = len(files)
        
        for i, f in enumerate(files):
            self.activity_update.emit(f"Membaca {title}: {f.name}")
            df_file = self.parse_cggtts_full(f)
            if not df_file.empty:
                raw_dfs.append(df_file)
            
            prog = start_prog + int(((i + 1) / total_files) * (end_prog - start_prog))
            self.progress_update.emit(prog)
            
        report = f"=== DATA {title.upper()} ===\n"
        report += f"Total File Terbaca : {total_files}\n"
        
        if not raw_dfs:
            report += "Tidak ditemukan data CGGTTS yang valid.\n\n"
            return report, pd.DataFrame()
            
        df_all = pd.concat(raw_dfs, ignore_index=True)
        df_grouped = df_all.groupby(['MJD', 'STTIME'], as_index=False)['REF'].mean()
        df_grouped = df_grouped.sort_values(['MJD', 'STTIME']).reset_index(drop=True)

        epoch = datetime(1858, 11, 17)
        unique_mjds = df_grouped['MJD'].unique()
        months_present = { (epoch + timedelta(days=int(m))).strftime('%B %Y') for m in unique_mjds }
            
        report += f"Bulan/Tahun Data   : {', '.join(sorted(list(months_present)))}\n"
        report += f"Total Hari (MJD)   : {len(unique_mjds)} hari\n"
        report += f"Total Titik Waktu  : {len(df_grouped)} titik\n"

        # Hitung Allan Variance
        y_vals = df_grouped['REF'].values
        if len(y_vals) > 2:
            allan_var = np.mean((y_vals[2:] - 2*y_vals[1:-1] + y_vals[:-2])**2) / 2
        else:
            allan_var = 0.0
        report += f"Allan Variance     : {allan_var:.6e} ns²\n\n"

        # Deteksi Jump >= 100 ns
        jumps = []
        for i in range(1, len(df_grouped)):
            diff = y_vals[i] - y_vals[i-1]
            if abs(diff) >= 100.0:
                jumps.append({
                    'MJD': df_grouped.loc[i, 'MJD'],
                    'STTIME': df_grouped.loc[i, 'STTIME'],
                    'Delta (ns)': diff
                })

        if jumps:
            report += f"STATUS JUMP: Ditemukan {len(jumps)} loncatan (>= 100 ns):\n"
            report += "-"*45 + "\n"
            report += f"{'No':<4} | {'MJD':<8} | {'STTIME':<8} | {'Besar Loncatan'}\n"
            report += "-"*45 + "\n"
            for idx, j in enumerate(jumps, 1):
                report += f"{idx:<4} | {j['MJD']:<8} | {j['STTIME']:<8} | {j['Delta (ns)']:.3f} ns\n"
        else:
            report += "STATUS JUMP: Tidak ada loncatan (jump) >= 100 ns terdeteksi.\n"

        report += "\n" + "="*45 + "\n\n"
        return report, df_grouped

    def run(self):
        try:
            self.progress_update.emit(0)
            
            report_std, self.df_std_grouped = self.analyze_folder_with_jump(self.dir_std, "Standard", 0, 50)
            report_uut, self.df_uut_grouped = self.analyze_folder_with_jump(self.dir_uut, "UUT", 50, 100)
            
            final_report = "LAPORAN CEK KELENGKAPAN & JUMP DETECTION DATA CGGTTS\n"
            final_report += "="*50 + "\n\n"
            final_report += report_std
            final_report += report_uut
            
            self.activity_update.emit("Pengecekan Selesai.")
            self.graph_data_update.emit(self.df_std_grouped, self.df_uut_grouped)
            self.report_update.emit(final_report)
            self.finished_update.emit("Selesai")
            
        except Exception as e:
            self.error_update.emit(str(e))


# ==========================================
# THREAD PEMROSESAN DATA 
# ==========================================
# ==========================================
# THREAD PEMROSESAN DATA 
# ==========================================
class AnalysisThread(QThread):
    activity_update = Signal(str)      
    progress_update = Signal(int)      
    finished_update = Signal(str)      
    error_update = Signal(str)
    report_update = Signal(str) 
    graph_data_update = Signal(object) 

    def __init__(self, dir_std, dir_uut, order_no):
        super().__init__()
        self.dir_std = dir_std
        self.dir_uut = dir_uut
        self.order_no = order_no
        self.total_matched = 0
        self.avg_diff = 0.0

    def parse_cggtts(self, filepath):
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            header_idx = -1
            header_cols = []
            for i, line in enumerate(lines):
                if 'MJD' in line and 'STTIME' in line:
                    header_idx = i
                    header_cols = line.strip().split()
                    break
            
            if header_idx == -1: return pd.DataFrame()

            ref_col_name = 'REFSYS' if 'REFSYS' in header_cols else 'REFGPS'
            sat_col_name = 'SAT' if 'SAT' in header_cols else 'PRN'
            
            idx_sat = header_cols.index(sat_col_name)
            idx_mjd = header_cols.index('MJD')
            idx_time = header_cols.index('STTIME')
            idx_ref = header_cols.index(ref_col_name)

            for line in lines[header_idx+2:]:
                parts = line.strip().split()
                if len(parts) > max(idx_sat, idx_mjd, idx_time, idx_ref):
                    sat = parts[idx_sat]
                    if not sat[0].isdigit(): sat = sat[1:] 
                    
                    data.append({
                        'SAT/PRN': int(sat),
                        'MJD': int(parts[idx_mjd]),
                        'STTIME': int(parts[idx_time]),
                        ref_col_name: float(parts[idx_ref]) / 10.0,
                        'File_Source': Path(filepath).name
                    })
        except: pass
        return pd.DataFrame(data)

    def stage_1_raw_extraction(self):
        self.activity_update.emit("Tahap 1: Mengekstrak data mentah...")
        std_list = [self.parse_cggtts(f) for f in Path(self.dir_std).glob('*') if f.is_file()]
        df_std_raw = pd.concat(std_list, ignore_index=True) if std_list else pd.DataFrame()
        
        uut_list = [self.parse_cggtts(f) for f in Path(self.dir_uut).glob('*') if f.is_file()]
        df_uut_raw = pd.concat(uut_list, ignore_index=True) if uut_list else pd.DataFrame()
        
        return df_std_raw, df_uut_raw

    def stage_2_grouping(self, df):
        self.activity_update.emit("Tahap 2: Grouping...")
        if df.empty: return pd.DataFrame()
        
        df_sorted = df.sort_values(['MJD', 'STTIME'])
        grouped_rows = []
        groups = df_sorted.groupby(['MJD', 'STTIME'])
        
        for _, frame in groups:
            for _, row in frame.iterrows():
                grouped_rows.append(row.to_dict())
            grouped_rows.append({k: None for k in df.columns})
            
        return pd.DataFrame(grouped_rows)

    def stage_3_matching(self, df_std, df_uut):
        self.activity_update.emit("Tahap 3: Sinkronisasi dan Format Kolom...")
        
        ref_std = 'REFSYS' if 'REFSYS' in df_std.columns else 'REFGPS'
        ref_uut = 'REFSYS' if 'REFSYS' in df_uut.columns else 'REFGPS'

        df_matched_raw = pd.merge(
            df_std, df_uut,
            on=['MJD', 'STTIME', 'SAT/PRN'],
            suffixes=('_STD', '_UUT')
        ).sort_values(['MJD', 'STTIME', 'SAT/PRN'])
        
        col_std = f"{ref_std}_STD" if f"{ref_std}_STD" in df_matched_raw.columns else ref_std
        col_uut = f"{ref_uut}_UUT" if f"{ref_uut}_UUT" in df_matched_raw.columns else ref_uut
        
        self.total_matched = len(df_matched_raw)

        formatted_rows = []
        groups = df_matched_raw.groupby(['MJD', 'STTIME'])
        
        for _, frame in groups:
            for _, row in frame.iterrows():
                formatted_rows.append({
                    'SAT/PRN_STD': row['SAT/PRN'],
                    'MJD_STD': row['MJD'],
                    'STTIME_STD': row['STTIME'],
                    f'{ref_std}_STD': row[col_std],
                    ' ': None,  
                    'SAT/PRN_UUT': row['SAT/PRN'],
                    'MJD_UUT': row['MJD'],
                    'STTIME_UUT': row['STTIME'],
                    f'{ref_uut}_UUT': row[col_uut]
                })
            formatted_rows.append({k: None for k in formatted_rows[-1].keys()})
            
        return df_matched_raw, pd.DataFrame(formatted_rows), ref_std, ref_uut, col_std, col_uut

    def get_bipm_corrections(self, unique_mjds):
        self.activity_update.emit("Tahap 4: Mengakses BIPM Circular T...")
        min_m = min(unique_mjds)
        max_m = max(unique_mjds)
        
        m1 = int(min_m)
        while not (str(m1).endswith('4') or str(m1).endswith('9')): m1 -= 1
        
        # Kita lebarkan batas atas hingga mjd + 1 agar data hari esok (untuk interpolasi intra-day) ikut tertarik jika memungkinkan
        m2 = int(max_m) + 1
        while not (str(m2).endswith('4') or str(m2).endswith('9')): m2 += 1
        
        url = f"https://webtai.bipm.org/api/v0.2-beta/get-data.html?scale=utc&lab=IDN&outfile=txt&mjd1={m1}&mjd2={m2}"
        self.activity_update.emit(f"Tahap 5: Mengunduh data BIPM (MJD {m1}-{m2})...")
        
        try:
            r = requests.get(url, timeout=15)
            bipm_data = {}
            for line in r.text.split('\n'):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    bipm_data[int(parts[0])] = float(parts[1])
            
            # Rentang MJD yang perlu dikalkulasi nilainya harian (termasuk MJD max + 1 untuk referensi hari esok)
            target_mjds = set(unique_mjds)
            for m in unique_mjds:
                target_mjds.add(m + 1)
            target_mjds = sorted(list(target_mjds))

            corrections = {}
            for m in target_mjds:
                if m in bipm_data:
                    corrections[m] = {'val': bipm_data[m], 'src': 'BIPM'}
                else:
                    lower_m = max([k for k in bipm_data.keys() if k < m], default=None)
                    upper_m = min([k for k in bipm_data.keys() if k > m], default=None)
                    
                    if lower_m is not None and upper_m is not None:
                        slope = (bipm_data[upper_m] - bipm_data[lower_m]) / (upper_m - lower_m)
                        interp_val = bipm_data[lower_m] + slope * (m - lower_m)
                        corrections[m] = {'val': round(interp_val, 3), 'src': 'Prediksi (Regresi)'}
                    elif lower_m is not None and upper_m is None:
                        past_keys = sorted([k for k in bipm_data.keys() if k < m], reverse=True)
                        if len(past_keys) >= 2:
                            l1 = past_keys[0]
                            l2 = past_keys[1]
                            slope = (bipm_data[l1] - bipm_data[l2]) / (l1 - l2)
                            extrap_val = bipm_data[l1] + slope * (m - l1)
                            corrections[m] = {'val': round(extrap_val, 3), 'src': 'Prediksi (Ekstrapolasi)'}
                        else:
                            corrections[m] = {'val': bipm_data[lower_m], 'src': 'Prediksi (Hold Terakhir)'}
                    else:
                        corrections[m] = {'val': 0.0, 'src': 'Tidak Ada Data API'}
            return corrections
        except Exception as e:
            self.error_update.emit(f"Gagal mengambil API BIPM: {e}")
            return {m: {'val': 0.0, 'src': 'Error API'} for m in unique_mjds}

    def get_time_interpolated_correction(self, mjd, sttime, bipm_corrections):
        # Mengambil nilai koreksi harian hari ini
        curr_info = bipm_corrections.get(mjd, {'val': 0.0, 'src': 'Tidak Ada'})
        val_today = curr_info['val']
        src = curr_info['src']
        
        # Mengambil nilai koreksi harian hari esok (mjd + 1)
        next_info = bipm_corrections.get(mjd + 1, None)
        if next_info is not None:
            val_tomorrow = next_info['val']
        else:
            val_tomorrow = val_today # Hold jika hari esok tidak tersedia
            
        # Konversi format STTIME (HHMMSS, contoh: 213015) menjadi total detik dalam sehari
        sttime_int = int(sttime)
        hours = sttime_int // 10000
        minutes = (sttime_int % 10000) // 100
        seconds = sttime_int % 100
        total_seconds = hours * 3600 + minutes * 60 + seconds
        
        # Interpolasi linier intra-day berdasarkan total detik dibagi 86400
        daily_diff = val_tomorrow - val_today
        fraction = float(total_seconds) / 86400.0
        interpolated_val = val_today + daily_diff * fraction
        
        return round(interpolated_val, 3), src

    def stage_4_correction(self, df_matched_raw, ref_std, ref_uut, col_std, col_uut):
        unique_mjds = df_matched_raw['MJD'].dropna().unique().tolist()
        if not unique_mjds:
            return pd.DataFrame(), {}
            
        bipm_corrections = self.get_bipm_corrections(unique_mjds)
        self.activity_update.emit("Tahap 6: Menerapkan Koreksi BIPM Per Waktu ke Standar...")
        
        formatted_rows = []
        groups = df_matched_raw.groupby(['MJD', 'STTIME'])
        
        for (mjd, sttime), frame in groups:
            koreksi, sumber = self.get_time_interpolated_correction(mjd, sttime, bipm_corrections)
            
            for _, row in frame.iterrows():
                refsys_std_val = row[col_std]
                refsys_uut_val = row[col_uut]
                
                formatted_rows.append({
                    'SAT/PRN_STD': row['SAT/PRN'],
                    'MJD_STD': mjd,
                    'STTIME_STD': sttime,
                    f'{ref_std}_STD': refsys_std_val,
                    'Nilai Koreksi': koreksi,
                    'Sumber Data': sumber,
                    f'{ref_std}_Terkoreksi': refsys_std_val + koreksi,
                    ' ': None,  
                    'SAT/PRN_UUT': row['SAT/PRN'],
                    'MJD_UUT': mjd,
                    'STTIME_UUT': sttime,
                    f'{ref_uut}_UUT': refsys_uut_val
                })
            formatted_rows.append({k: None for k in formatted_rows[-1].keys()})
            
        return pd.DataFrame(formatted_rows), bipm_corrections

    def stage_5_difference(self, df_matched_raw, ref_std, ref_uut, col_std, col_uut, bipm_corrections):
        self.activity_update.emit("Tahap 7: Menghitung Selisih (STD Terkoreksi - UUT)...")
        
        formatted_rows = []
        groups = df_matched_raw.groupby(['MJD', 'STTIME'])
        
        total_diff = 0
        count = 0

        for (mjd, sttime), frame in groups:
            koreksi, sumber = self.get_time_interpolated_correction(mjd, sttime, bipm_corrections)
            
            for _, row in frame.iterrows():
                refsys_std_val = row[col_std]
                refsys_uut_val = row[col_uut]
                
                refsys_terkoreksi = refsys_std_val + koreksi
                selisih = refsys_terkoreksi - refsys_uut_val
                
                total_diff += selisih
                count += 1

                formatted_rows.append({
                    'SAT/PRN_STD': row['SAT/PRN'],
                    'MJD_STD': mjd,
                    'STTIME_STD': sttime,
                    f'{ref_std}_STD': refsys_std_val,
                    'Nilai Koreksi': koreksi,
                    'Sumber Data': sumber,
                    f'{ref_std}_Terkoreksi': refsys_terkoreksi,
                    ' ': None,  
                    'SAT/PRN_UUT': row['SAT/PRN'],
                    'MJD_UUT': mjd,
                    'STTIME_UUT': sttime,
                    f'{ref_uut}_UUT': refsys_uut_val,
                    '  ': None, 
                    'Selisih (ns)': selisih
                })
            formatted_rows.append({k: None for k in formatted_rows[-1].keys()})
        
        if count > 0:
            self.avg_diff = total_diff / count

        return pd.DataFrame(formatted_rows)

    def stage_6_time_average(self, df_matched_raw, ref_std, ref_uut, col_std, col_uut, bipm_corrections):
        self.activity_update.emit("Tahap 8: Menghitung Rata-rata per Waktu (STTIME)...")
        
        time_rows = []
        groups_mjd = df_matched_raw.groupby('MJD')
        
        for mjd, frame_mjd in groups_mjd:
            groups_sttime = frame_mjd.groupby('STTIME')
            
            for sttime, frame in groups_sttime:
                koreksi, _ = self.get_time_interpolated_correction(mjd, sttime, bipm_corrections)
                mean_std_raw = frame[col_std].mean()
                mean_uut = frame[col_uut].mean()
                
                mean_std_terkoreksi = mean_std_raw + koreksi
                mean_selisih = mean_std_terkoreksi - mean_uut
                
                time_rows.append({
                    'MJD': mjd,
                    'STTIME': sttime,
                    f'Rata-rata {ref_std}_STD Terkoreksi': mean_std_terkoreksi,
                    f'Rata-rata {ref_uut}_UUT': mean_uut,
                    'Rata-rata Selisih (ns)': mean_selisih
                })
            time_rows.append({k: None for k in time_rows[-1].keys()})
            
        return pd.DataFrame(time_rows)

    def stage_7_report(self, df_std_raw, df_uut_raw, df_matched_raw, df_tahap_6, ref_std, ref_uut, col_std, col_uut, bipm_corrections):
        self.activity_update.emit("Tahap 9: Membuat Report Analisis...")
        
        if not df_matched_raw.empty:
            mode_mjd = df_matched_raw['MJD'].mode()[0]
            epoch = datetime(1858, 11, 17)
            mode_date = epoch + timedelta(days=float(mode_mjd))
            month_year = mode_date.strftime('%B %Y')
        else:
            month_year = "Unknown"
            
        jml_std = len(df_std_raw)
        jml_uut = len(df_uut_raw)
        jml_match = len(df_matched_raw)

        df_valid = df_tahap_6.dropna(subset=['MJD']).copy()
        if len(df_valid) > 1:
            x = np.arange(len(df_valid))
            y_std = df_valid[f'Rata-rata {ref_std}_STD Terkoreksi'].astype(float).values
            y_uut = df_valid[f'Rata-rata {ref_uut}_UUT'].astype(float).values
            y_diff = df_valid['Rata-rata Selisih (ns)'].astype(float).values
            
            slope_std, _ = np.polyfit(x, y_std, 1)
            slope_uut, _ = np.polyfit(x, y_uut, 1)
            
            if len(y_diff) > 2:
                allan_var = np.mean((y_diff[2:] - 2*y_diff[1:-1] + y_diff[:-2])**2) / 2
            else:
                allan_var = 0.0
        else:
            slope_std = slope_uut = allan_var = 0.0

        daily_rows = []
        groups_harian = df_matched_raw.groupby('MJD')
        for mjd, frame in groups_harian:
            mean_sttime = frame['STTIME'].mean() if 'STTIME' in frame.columns else 0
            koreksi, _ = self.get_time_interpolated_correction(mjd, mean_sttime, bipm_corrections)
            
            mean_std_raw = frame[col_std].mean()
            mean_uut = frame[col_uut].mean()
            mean_std_terkoreksi = mean_std_raw + koreksi
            mean_selisih = mean_std_terkoreksi - mean_uut
            
            daily_rows.append({
                'MJD': mjd,
                f'Rata {ref_std}_STD Terkoreksi (ns)': round(mean_std_terkoreksi, 3), 
                f'Rata {ref_uut}_UUT (ns)': round(mean_uut, 3),                       
                'Rata Selisih (ns)': round(mean_selisih, 3)                           
            })
        df_daily = pd.DataFrame(daily_rows)

        report_text = (
            f"=== REPORT ANALISIS KALIBRASI ===\n\n"
            f"Bulan dan Tahun Analisis : {month_year}\n"
            f"Jumlah Data Standard     : {jml_std}\n"
            f"Jumlah Data UUT          : {jml_uut}\n"
            f"Jumlah Data Matched      : {jml_match}\n"
            f"Regresi STD Terkoreksi   : {slope_std:.6f} ns/point\n" 
            f"Regresi UUT              : {slope_uut:.6f} ns/point\n"
            f"Allan Variance Akhir     : {allan_var:.6f} ns²\n"
            f"---------------------------------\n\n"
            f"TABEL RATA-RATA HARIAN\n"
        )
        report_text += df_daily.to_string(index=False)

        excel_rows = [
            {'Kolom A': 'Bulan dan Tahun Analisis', 'Kolom B': month_year, 'Kolom C': '', 'Kolom D': ''},
            {'Kolom A': 'Jumlah Data Standard', 'Kolom B': jml_std, 'Kolom C': '', 'Kolom D': ''},
            {'Kolom A': 'Jumlah Data UUT', 'Kolom B': jml_uut, 'Kolom C': '', 'Kolom D': ''},
            {'Kolom A': 'Jumlah Data Matched', 'Kolom B': jml_match, 'Kolom C': '', 'Kolom D': ''},
            {'Kolom A': 'Regresi STD Terkoreksi (ns/point)', 'Kolom B': slope_std, 'Kolom C': '', 'Kolom D': ''}, 
            {'Kolom A': 'Regresi UUT (ns/point)', 'Kolom B': slope_uut, 'Kolom C': '', 'Kolom D': ''},
            {'Kolom A': 'Allan Variance (ns²)', 'Kolom B': allan_var, 'Kolom C': '', 'Kolom D': ''},
            {'Kolom A': '', 'Kolom B': '', 'Kolom C': '', 'Kolom D': ''},
            {'Kolom A': 'MJD', 'Kolom B': f'Rata-rata {ref_std}_STD Terkoreksi (ns)', 'Kolom C': f'Rata-rata {ref_uut}_UUT (ns)', 'Kolom D': 'Rata-rata Selisih (ns)'}
        ]
        
        for _, row in df_daily.iterrows():
            excel_rows.append({
                'Kolom A': row['MJD'],
                'Kolom B': row[f'Rata {ref_std}_STD Terkoreksi (ns)'],
                'Kolom C': row[f'Rata {ref_uut}_UUT (ns)'],
                'Kolom D': row['Rata Selisih (ns)']
            })
            
        df_report_excel = pd.DataFrame(excel_rows)

        return df_report_excel, report_text

    def run(self):
        try:
            self.progress_update.emit(10)
            
            df_std_raw, df_uut_raw = self.stage_1_raw_extraction()
            if df_std_raw.empty or df_uut_raw.empty:
                self.error_update.emit("Data tidak ditemukan.")
                return
            
            self.progress_update.emit(20)
            
            self.activity_update.emit("Tahap 2: Mengelompokkan data per waktu...")
            df_std_grouped = self.stage_2_grouping(df_std_raw)
            df_uut_grouped = self.stage_2_grouping(df_uut_raw)
            
            self.progress_update.emit(35)
            
            df_matched_raw, df_matched_formatted, ref_std, ref_uut, col_std, col_uut = self.stage_3_matching(df_std_raw, df_uut_raw)
            
            self.progress_update.emit(50)

            df_tahap_4, bipm_corrections = self.stage_4_correction(df_matched_raw, ref_std, ref_uut, col_std, col_uut)
            
            self.progress_update.emit(65)

            df_tahap_5 = self.stage_5_difference(df_matched_raw, ref_std, ref_uut, col_std, col_uut, bipm_corrections)

            self.progress_update.emit(75)
            
            df_tahap_6 = self.stage_6_time_average(df_matched_raw, ref_std, ref_uut, col_std, col_uut, bipm_corrections)

            self.progress_update.emit(85)

            df_tahap_7_report, text_report = self.stage_7_report(
                df_std_raw, df_uut_raw, df_matched_raw, df_tahap_6, ref_std, ref_uut, col_std, col_uut, bipm_corrections
            )

            self.activity_update.emit("Tahap Akhir: Menulis file Excel...")
            output_path = Path(self.dir_uut).parent / f"Analisis_{self.order_no.replace('/','-')}.xlsx"
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_std_raw.to_excel(writer, sheet_name="T1_Raw_Standard", index=False)
                df_uut_raw.to_excel(writer, sheet_name="T1_Raw_UUT", index=False)
                df_std_grouped.to_excel(writer, sheet_name="T2_Grouped_Standard", index=False)
                df_uut_grouped.to_excel(writer, sheet_name="T2_Grouped_UUT", index=False)
                df_matched_formatted.to_excel(writer, sheet_name="T3_Matched_Analysis", index=False)
                df_tahap_4.to_excel(writer, sheet_name="T4_Correction_Standar", index=False)
                df_tahap_5.to_excel(writer, sheet_name="T5_Selisih_Akhir", index=False)
                df_tahap_6.to_excel(writer, sheet_name="T6_Rata_Waktu", index=False) 
                df_tahap_7_report.to_excel(writer, sheet_name="T7_Report_Akhir", index=False, header=False) 

            self.progress_update.emit(100)
            self.activity_update.emit("Proses Selesai.")
            
            summary = (f"ANALISIS SELESAI\n"
                       f"---------------------------\n"
                       f"No. Order : {self.order_no}\n"
                       f"Total Data Terproses : {self.total_matched} baris sinkron\n"
                       f"Rata-rata Selisih : {self.avg_diff:.3f} ns\n"
                       f"File Output : Analisis_{self.order_no.replace('/','-')}.xlsx\n"
                       f"Lokasi : {output_path.parent}")
            
            self.finished_update.emit(summary)
            self.report_update.emit(text_report) 
            self.graph_data_update.emit(df_tahap_6) 

        except Exception as e:
            self.error_update.emit(str(e))

# ==========================================
# GUI UTAMA
# ==========================================
class CGGTTSMainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.font_main = QFont("Segoe UI", 10)
        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        self.setWindowTitle('SNSU-BSN CGGTTS Analyzer')
        self.setWindowIcon(QIcon(resource_path('logo.ico')))
        self.resize(1000, 700)
        
        self.setStyleSheet("""
            QWidget { background-color: white; font-family: 'Segoe UI'; color: black;}
            QPushButton { background-color: #388e3c; color: white; border-radius: 4px; padding: 6px; font-weight: bold; border: 1px solid #2e7d32;}
            QPushButton:hover { background-color: #2e7d32; }
            QPushButton:disabled { background-color: #a5d6a7; color: #e8f5e9; }
            QLineEdit { background-color: white; border: 1px solid #a5d6a7; border-radius: 4px; padding: 5px; color: black; }
            QProgressBar { border: 1px solid #a5d6a7; border-radius: 4px; text-align: center; background-color: white; color: black; font-weight: bold;}
            QProgressBar::chunk { background-color: #4caf50; border-radius: 3px;}
            QTabWidget::pane { border: 1px solid #a5d6a7; background: white; border-radius: 4px;}
            QTabBar::tab { background: #c8e6c9; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px;}
            QTabBar::tab:selected { background: white; border-bottom-color: white; font-weight: bold; }
            QTextEdit { background-color: #fafafa; border: 1px solid #c8e6c9; color: black; }
        """)

        main_layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        
        title_layout = QVBoxLayout()
        lbl_title = QLabel("CGGTTS Analyser ver. 2.1")
        lbl_title.setStyleSheet("font-size: 13pt; font-weight: bold; color: black;")
        
        lbl_sub = QLabel("National Measurement Standard\nNational Standardization Agency of Indonesia")
        lbl_sub.setStyleSheet("font-size: 13pt; font-weight: bold; color: black;") 
        
        title_layout.addWidget(lbl_title)
        title_layout.addWidget(lbl_sub)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        self.lbl_clock = QLabel()
        self.lbl_clock.setStyleSheet("font-weight: bold; color: black; background: white; padding: 10px; border-radius: 5px; border: 1px solid #c8e6c9;")
        header_layout.addWidget(self.lbl_clock)
        
        main_layout.addLayout(header_layout)

        self.tabs = QTabWidget()
        
        self.tab_cek_data = QWidget()
        self.setup_cek_data_tab()
        self.tabs.addTab(self.tab_cek_data, "Cek Kelengkapan Data")

        self.tab_remote = QWidget()
        self.setup_remote_tab()
        self.tabs.addTab(self.tab_remote, "Remote Clock Calibration")

        self.tab_kalkulator_mjd = QWidget()
        self.setup_kalkulator_mjd_tab()
        self.tabs.addTab(self.tab_kalkulator_mjd, "Kalkulator MJD")

        self.tab_utc = QWidget()
        self.setup_utc_tab()
        self.tabs.addTab(self.tab_utc, "Analisis UTC(K)")

        self.tab_mc = QWidget()
        self.setup_monte_carlo_tab()
        self.tabs.addTab(self.tab_mc, "Analisis Monte Carlo")

        self.tab_informasi = QWidget()
        self.setup_informasi_tab()
        self.tabs.addTab(self.tab_informasi, "Informasi")

        self.tabs.setCurrentIndex(0) 
        main_layout.addWidget(self.tabs)

        lbl_footer = QLabel("CGGTTS Analyser for cesium atomic clock calibration v.2.1 © SNSU Time and Frequency 2026")
        lbl_footer.setAlignment(Qt.AlignLeft)
        lbl_footer.setStyleSheet("font-size: 8pt; color: black; margin-top: 5px;")
        main_layout.addWidget(lbl_footer)

    # ==========================================
    # FUNGSI SETUP TAB KHUSUS MONTE CARLO
    # ==========================================
    def setup_monte_carlo_tab(self):
        tab_layout = QVBoxLayout(self.tab_mc)
        tab_layout.setContentsMargins(15, 20, 15, 15)

        h_top = QHBoxLayout()
        h_top.addWidget(QLabel("Jumlah Sumber Ketidakpastian:"))
        self.txt_mc_count = QLineEdit("5")
        self.btn_mc_gen = QPushButton("Buat Tabel", clicked=self.generate_mc_table)
        h_top.addWidget(self.txt_mc_count)
        h_top.addWidget(self.btn_mc_gen)
        
        h_top.addSpacing(20)
        h_top.addWidget(QLabel("Jumlah Random Sample (N):"))
        self.txt_mc_n = QLineEdit("200000")
        h_top.addWidget(self.txt_mc_n)
        h_top.addStretch()
        
        tab_layout.addLayout(h_top)

        self.table_mc = QTableWidget(0, 3)
        self.table_mc.setHorizontalHeaderLabels(["Sumber Ketidakpastian", "Distribusi", "Nilai u"])
        self.table_mc.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tab_layout.addWidget(self.table_mc)

        bottom_layout = QHBoxLayout()
        left_side = QVBoxLayout()
        
        self.btn_mc_start = QPushButton("JALANKAN MONTE CARLO", clicked=self.run_mc_process)
        self.btn_mc_start.setFixedHeight(45)
        self.btn_mc_start.setStyleSheet("background-color: #ff9800; border: 1px solid #ef6c00;")
        
        self.btn_mc_graph = QPushButton("TAMPILKAN GRAFIK DISTRIBUSI", clicked=self.show_mc_graph)
        self.btn_mc_graph.setFixedHeight(35)
        self.btn_mc_graph.setStyleSheet("background-color: #1976d2; border: 1px solid #1565c0;")
        self.btn_mc_graph.setEnabled(False) 
        
        self.lbl_activity_mc = QLabel("Menunggu input...")
        self.lbl_activity_mc.setStyleSheet("color: #d84315; font-style: italic; font-weight: bold;")
        
        left_side.addWidget(self.btn_mc_start)
        left_side.addWidget(self.btn_mc_graph)
        left_side.addSpacing(10)
        left_side.addWidget(self.lbl_activity_mc)
        left_side.addStretch()

        right_side = QVBoxLayout()
        right_side.addWidget(QLabel("<b>HASIL ANALISIS:</b>"))
        self.log_mc = QTextEdit(readOnly=True)
        self.log_mc.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        self.log_mc.setPlaceholderText("Laporan ringkasan Monte Carlo akan ditampilkan di sini...")
        right_side.addWidget(self.log_mc)

        bottom_layout.addLayout(left_side, 1)
        bottom_layout.addLayout(right_side, 2)
        tab_layout.addLayout(bottom_layout)
        
        self.generate_mc_table()

    def generate_mc_table(self):
        try:
            count = int(self.txt_mc_count.text())
        except ValueError:
            return
        self.table_mc.setRowCount(count)
        for row in range(count):
            self.table_mc.setItem(row, 0, QTableWidgetItem(f"Sumber {row+1}"))
            cb = QComboBox()
            cb.addItems(["Normal", "Rectangular"])
            self.table_mc.setCellWidget(row, 1, cb)
            self.table_mc.setItem(row, 2, QTableWidgetItem("0.0"))

    def run_mc_process(self):
        try:
            N = int(self.txt_mc_n.text())
        except:
            self.lbl_activity_mc.setText("Error: Nilai N harus integer!")
            return
            
        inputs_data = []
        for row in range(self.table_mc.rowCount()):
            name_item = self.table_mc.item(row, 0)
            name = name_item.text() if name_item else f"Sumber {row+1}"
            
            cb = self.table_mc.cellWidget(row, 1)
            dist = cb.currentText() if cb else "Normal"
            if dist == "Rectangular":
                dist = "rect"
            
            u_item = self.table_mc.item(row, 2)
            try:
                u_val = float(u_item.text()) if u_item else 0.0
            except:
                u_val = 0.0
            
            inputs_data.append((name, dist, u_val))

        self.btn_mc_start.setEnabled(False)
        self.btn_mc_graph.setEnabled(False)
        self.log_mc.clear()

        self.mc_worker = MonteCarloThread(inputs_data, N)
        self.mc_worker.activity_update.connect(lambda m: self.lbl_activity_mc.setText(f"Aktifitas: {m}"))
        self.mc_worker.report_update.connect(self.log_mc.setPlainText)
        self.mc_worker.graph_data_update.connect(self.receive_mc_graph)
        self.mc_worker.finished_update.connect(self.on_mc_finish)
        self.mc_worker.error_update.connect(self.on_mc_err)
        self.mc_worker.start()

    def receive_mc_graph(self, Y):
        self.mc_Y = Y

    def on_mc_finish(self, msg):
        self.btn_mc_start.setEnabled(True)
        self.btn_mc_graph.setEnabled(True)
        self.lbl_activity_mc.setText("Selesai")

    def on_mc_err(self, msg):
        self.btn_mc_start.setEnabled(True)
        self.lbl_activity_mc.setText("Error terjadi")
        self.log_mc.setPlainText(f"[ERROR]\n{msg}")

    def show_mc_graph(self):
        if not hasattr(self, 'mc_Y'):
            return
        plt.figure("Grafik Distribusi Monte Carlo", figsize=(8, 5))
        plt.hist(self.mc_Y, bins=100, density=True, alpha=0.7, color='g')
        plt.title("Distribusi Gabungan Monte Carlo")
        plt.xlabel("Nilai")
        plt.ylabel("Frekuensi")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ==========================================
    # FUNGSI SETUP TAB KHUSUS ANALISIS UTC(K)
    # ==========================================
    def setup_utc_tab(self):
        tab_layout = QVBoxLayout(self.tab_utc)
        tab_layout.setContentsMargins(15, 20, 15, 15)

        input_box = QGroupBox("Pengaturan API BIPM")
        input_box.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #a5d6a7; border-radius: 5px; margin-top: 10px; padding-top: 15px; font-size: 11pt; }")
        
        form_layout = QHBoxLayout()
        
        v_param = QVBoxLayout()
        v_param.addWidget(QLabel("MJD Awal:"))
        self.txt_utc_start = QLineEdit()
        self.txt_utc_start.setPlaceholderText("Contoh: 60000")
        v_param.addWidget(self.txt_utc_start)
        
        v_param.addWidget(QLabel("MJD Akhir:"))
        self.txt_utc_stop = QLineEdit()
        self.txt_utc_stop.setPlaceholderText("Contoh: 60050")
        v_param.addWidget(self.txt_utc_stop)

        v_param.addWidget(QLabel("Kode Lab (UTC(K)):"))
        self.txt_utc_lab = QLineEdit()
        self.txt_utc_lab.setText("IDN")
        v_param.addWidget(self.txt_utc_lab)
        
        v_out = QVBoxLayout()
        v_out.addWidget(QLabel("Lokasi Simpan Excel (Opsional):"))
        self.txt_utc_save = QLineEdit(readOnly=True)
        self.txt_utc_save.setPlaceholderText("Pilih lokasi simpan...")
        btn_save = QPushButton("Pilih Lokasi", clicked=self.get_save_path_utc)
        
        h_save = QHBoxLayout()
        h_save.addWidget(self.txt_utc_save)
        h_save.addWidget(btn_save)
        v_out.addLayout(h_save)
        v_out.addStretch()

        form_layout.addLayout(v_param, 1)
        form_layout.addSpacing(20)
        form_layout.addLayout(v_out, 2)
        
        input_box.setLayout(form_layout)
        tab_layout.addWidget(input_box)

        bottom_layout = QHBoxLayout()
        
        left_side = QVBoxLayout()
        self.btn_utc_start = QPushButton("AMBIL DATA & ANALISIS", clicked=self.run_utc_process)
        self.btn_utc_start.setFixedHeight(45)
        self.btn_utc_start.setStyleSheet("background-color: #ff9800; border: 1px solid #ef6c00;")
        
        self.btn_utc_graph = QPushButton("TAMPILKAN GRAFIK", clicked=self.show_utc_graph)
        self.btn_utc_graph.setFixedHeight(35)
        self.btn_utc_graph.setStyleSheet("background-color: #1976d2; border: 1px solid #1565c0;")
        self.btn_utc_graph.setEnabled(False) 

        self.lbl_activity_utc = QLabel("Menunggu input...")
        self.lbl_activity_utc.setStyleSheet("color: #d84315; font-style: italic; font-weight: bold;")
        
        left_side.addWidget(self.btn_utc_start)
        left_side.addWidget(self.btn_utc_graph)
        left_side.addSpacing(10)
        left_side.addWidget(self.lbl_activity_utc)
        left_side.addStretch()

        right_side = QVBoxLayout()
        right_side.addWidget(QLabel("<b>HASIL ANALISIS:</b>"))
        self.log_utc = QTextEdit(readOnly=True) 
        self.log_utc.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        self.log_utc.setPlaceholderText("Laporan regresi linier akan ditampilkan di sini...")
        right_side.addWidget(self.log_utc)

        bottom_layout.addLayout(left_side, 1)
        bottom_layout.addLayout(right_side, 2)

        tab_layout.addLayout(bottom_layout)

    def get_save_path_utc(self):
        path, _ = QFileDialog.getSaveFileName(self, "Simpan File Excel", "", "Excel Files (*.xlsx)")
        if path:
            self.txt_utc_save.setText(path)

    def run_utc_process(self):
        if not self.txt_utc_start.text() or not self.txt_utc_stop.text() or not self.txt_utc_lab.text():
            self.lbl_activity_utc.setText("Error: Harap isi MJD Awal, Akhir, dan Kode Lab!")
            return
            
        self.btn_utc_start.setEnabled(False)
        self.btn_utc_graph.setEnabled(False)
        self.log_utc.clear()
        
        self.utc_worker = UTCAnalysisThread(
            self.txt_utc_start.text(), 
            self.txt_utc_stop.text(), 
            self.txt_utc_lab.text(), 
            self.txt_utc_save.text()
        )
        self.utc_worker.activity_update.connect(lambda m: self.lbl_activity_utc.setText(f"Aktifitas: {m}"))
        self.utc_worker.report_update.connect(self.log_utc.setPlainText) 
        self.utc_worker.graph_data_update.connect(self.receive_utc_graph_data) 
        self.utc_worker.error_update.connect(self.on_utc_err)
        self.utc_worker.finished_update.connect(self.on_utc_finish)
        self.utc_worker.start()

    def receive_utc_graph_data(self, x, y, reg_vals, eq_str):
        self.utc_x = x
        self.utc_y = y
        self.utc_reg_vals = reg_vals
        self.utc_eq_str = eq_str

    def on_utc_finish(self, msg):
        self.btn_utc_start.setEnabled(True)
        self.btn_utc_graph.setEnabled(True)
        self.lbl_activity_utc.setText("Selesai")

    def on_utc_err(self, msg):
        self.btn_utc_start.setEnabled(True)
        self.lbl_activity_utc.setText("Error terjadi")
        self.log_utc.setPlainText(f"[ERROR]\n{msg}")

    def show_utc_graph(self):
        if not hasattr(self, 'utc_x'):
            return
        plt.figure("Grafik Regresi Linier UTC(K)", figsize=(9, 6))
        plt.plot(self.utc_x, self.utc_y, 'b.', label='Data Asli')
        plt.plot(self.utc_x, self.utc_reg_vals, 'r-', label=self.utc_eq_str)
        plt.title(f"Analisis UTC - UTC({self.txt_utc_lab.text()})")
        plt.xlabel("MJD")
        plt.ylabel("Waktu (ns)")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ==========================================
    # TAB INFORMASI & TAB LAINNYA
    # ==========================================
    def setup_informasi_tab(self):
        layout = QVBoxLayout(self.tab_informasi)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        info_text = (
            "<b>Informasi Aplikasi:</b><br>"
            "Versi &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 2.1<br>"
            "Developer &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Reggi Aryunadi<br>"
            "Hak Cipta/Lisensi : SNSU-BSN (Internal used only)<br>"
            "Update &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Mei 2026<br><br>"
            "<b>Daftar Fitur:</b><br>"
            "- Cek Kelengkapan Data CGGTTS<br>"
            "- Remote Clock Calibration<br>"
            "- Jump Detection<br>"
            "- Kalkulator MJD<br>"
            "- Analisis Regresi UTC(K)<br>"
            "- Analisis Ketidakpastian Monte Carlo<br><br>"
            "<b>Versi CGGTTS yang didukung:</b><br>"
            "- CGGTTS Versi 01<br>"
            "- CGGTTS Versi 2E"
        )
        lbl_info = QLabel(info_text)
        lbl_info.setFont(self.font_main)
        layout.addWidget(lbl_info)

        btn_manual = QPushButton("Buka Manual Book (PDF)")
        btn_manual.setFixedWidth(250)
        btn_manual.setStyleSheet("background-color: #00796b; border: 1px solid #004d40;")
        btn_manual.clicked.connect(self.open_manual_book)
        layout.addWidget(btn_manual)

        grp_feedback = QGroupBox("Keluhan atau Bantuan")
        grp_feedback.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #a5d6a7; border-radius: 5px; margin-top: 10px; padding-top: 15px; font-size: 10pt; }")
        v_feed = QVBoxLayout()
        
        lbl_feed = QLabel("Silakan tulis keluhan, bug, atau permintaan bantuan di bawah ini. Pesan akan disimpan untuk dibaca oleh developer.")
        lbl_feed.setWordWrap(True)
        v_feed.addWidget(lbl_feed)

        self.txt_feedback = QTextEdit()
        self.txt_feedback.setFixedHeight(80)
        self.txt_feedback.setPlaceholderText("Ketik pesan Anda di sini...")
        v_feed.addWidget(self.txt_feedback)

        btn_submit_feedback = QPushButton("Kirim / Simpan Pesan")
        btn_submit_feedback.setFixedWidth(180)
        btn_submit_feedback.clicked.connect(self.save_feedback)
        v_feed.addWidget(btn_submit_feedback)

        grp_feedback.setLayout(v_feed)
        layout.addWidget(grp_feedback)
        layout.addStretch()

    def open_manual_book(self):
        pdf_path = Path(resource_path("cggtts analyzer operation manual.pdf")).resolve()
        if pdf_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path)))
        else:
            QMessageBox.warning(self, "Peringatan", "File 'cggtts analyzer operation manual .pdf' tidak ditemukan di folder yang sama!")

    def save_feedback(self):
        pesan = self.txt_feedback.toPlainText().strip()
        if not pesan:
            return
        
        try:
            with open("keluhan_bantuan_log.txt", "a", encoding="utf-8") as f:
                waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{waktu}]\n{pesan}\n")
                f.write("-" * 40 + "\n")
            
            self.txt_feedback.clear()
            QMessageBox.information(self, "Sukses", "Pesan Anda telah berhasil disimpan untuk developer.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menyimpan pesan: {e}")

    def setup_cek_data_tab(self):
        tab_layout = QVBoxLayout(self.tab_cek_data)
        tab_layout.setContentsMargins(15, 20, 15, 15)

        input_box = QHBoxLayout()
        v_btns = QVBoxLayout()
        v_edits = QVBoxLayout()
        
        v_btns.addWidget(QPushButton("Directory Standard", clicked=lambda: self.get_path(self.txt_std_check)))
        v_btns.addWidget(QPushButton("Directory UUT", clicked=lambda: self.get_path(self.txt_uut_check)))

        self.txt_std_check = QLineEdit(readOnly=True)
        self.txt_std_check.setPlaceholderText("Pilih folder standar...")
        self.txt_uut_check = QLineEdit(readOnly=True)
        self.txt_uut_check.setPlaceholderText("Pilih folder UUT...")

        v_edits.addWidget(self.txt_std_check)
        v_edits.addWidget(self.txt_uut_check)

        input_box.addLayout(v_btns, 1)
        input_box.addLayout(v_edits, 4)
        tab_layout.addLayout(input_box)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)

        left_side = QVBoxLayout()
        self.btn_check = QPushButton("CEK KELENGKAPAN", clicked=self.run_check_data)
        self.btn_check.setFixedHeight(45)
        self.btn_check.setStyleSheet("background-color: #ff9800; border: 1px solid #ef6c00;")
        self.btn_check_graph = QPushButton("TAMPILKAN GRAFIK", clicked=self.show_check_graph)
        self.btn_check_graph.setFixedHeight(35)
        self.btn_check_graph.setStyleSheet("background-color: #1976d2; border: 1px solid #1565c0;")
        self.btn_check_graph.setEnabled(False)
        
        self.pbar_check = QProgressBar()
        self.pbar_check.setFixedHeight(25)
        
        self.lbl_activity_check = QLabel("Menunggu input...")
        self.lbl_activity_check.setStyleSheet("color: #d84315; font-style: italic; font-weight: bold;")
        
        left_side.addWidget(self.btn_check)
        left_side.addWidget(self.btn_check_graph)
        left_side.addSpacing(10)
        left_side.addWidget(QLabel("Progress:"))
        left_side.addWidget(self.pbar_check)
        left_side.addWidget(self.lbl_activity_check)
        left_side.addStretch()

        right_side = QVBoxLayout()
        right_side.addWidget(QLabel("<b>HASIL CEK KELENGKAPAN DATA:</b>"))
        self.log_check = QTextEdit(readOnly=True) 
        self.log_check.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        self.log_check.setPlaceholderText("Laporan kelengkapan data (Bulan, Hari, STTIME) akan ditampilkan di sini...")
        right_side.addWidget(self.log_check)

        bottom_layout.addLayout(left_side, 1)  
        bottom_layout.addLayout(right_side, 2) 
        
        tab_layout.addLayout(bottom_layout)

    def setup_kalkulator_mjd_tab(self):
        layout = QVBoxLayout(self.tab_kalkulator_mjd)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        info_lbl = QLabel(
            "<b>Modified Julian Date (MJD)</b> adalah standar representasi waktu dalam metrologi dan astronomi.<br>"
            "MJD 0 dihitung sejak 17 November 1858, pukul 00:00:00 UTC."
        )
        info_lbl.setStyleSheet("background-color: #e8f5e9; padding: 15px; border-radius: 5px; border: 1px solid #c8e6c9; color: #2e7d32; font-size: 11pt;")
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        split_layout = QHBoxLayout()
        split_layout.setSpacing(20)

        grp_date = QGroupBox("Konversi Tanggal ke MJD")
        grp_date.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #a5d6a7; border-radius: 5px; margin-top: 10px; padding-top: 15px; font-size: 11pt; }")
        v_date = QVBoxLayout()
        v_date.setSpacing(15)
        
        self.date_input = QDateTimeEdit(QDateTime.currentDateTimeUtc())
        self.date_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.date_input.setStyleSheet("padding: 8px; font-size: 11pt;")
        
        btn_to_mjd = QPushButton("Hitung Nilai MJD")
        btn_to_mjd.clicked.connect(self.calc_mjd_from_date)
        
        self.lbl_out_mjd = QLabel("MJD:\n-")
        self.lbl_out_mjd.setAlignment(Qt.AlignCenter)
        self.lbl_out_mjd.setStyleSheet("font-size: 14pt; font-weight: bold; color: #1565c0; padding: 15px; background: #e3f2fd; border-radius: 5px;")
        
        v_date.addWidget(QLabel("Pilih Tanggal dan Waktu (Format UTC):"))
        v_date.addWidget(self.date_input)
        v_date.addWidget(btn_to_mjd)
        v_date.addWidget(self.lbl_out_mjd)
        v_date.addStretch()
        grp_date.setLayout(v_date)

        grp_mjd = QGroupBox("Konversi MJD ke Tanggal")
        grp_mjd.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #a5d6a7; border-radius: 5px; margin-top: 10px; padding-top: 15px; font-size: 11pt; }")
        v_mjd = QVBoxLayout()
        v_mjd.setSpacing(15)
        
        self.mjd_input = QLineEdit()
        self.mjd_input.setPlaceholderText("Masukkan nilai MJD (contoh: 61074)")
        self.mjd_input.setStyleSheet("padding: 8px; font-size: 11pt;")
        
        btn_to_date = QPushButton("Hitung Tanggal")
        btn_to_date.clicked.connect(self.calc_date_from_mjd)
        
        self.lbl_out_date = QLabel("Waktu UTC:\n-\n\nWaktu WIB:\n-")
        self.lbl_out_date.setAlignment(Qt.AlignCenter)
        self.lbl_out_date.setStyleSheet("font-size: 12pt; font-weight: bold; color: #e65100; padding: 15px; background: #fff3e0; border-radius: 5px;")
        
        v_mjd.addWidget(QLabel("Masukkan Nilai MJD (Dapat berupa desimal):"))
        v_mjd.addWidget(self.mjd_input)
        v_mjd.addWidget(btn_to_date)
        v_mjd.addWidget(self.lbl_out_date)
        v_mjd.addStretch()
        grp_mjd.setLayout(v_mjd)

        split_layout.addWidget(grp_date)
        split_layout.addWidget(grp_mjd)
        layout.addLayout(split_layout)
        layout.addStretch()

    def calc_mjd_from_date(self):
        try:
            dt = self.date_input.dateTime().toPython()
            dt_utc = dt.replace(tzinfo=timezone.utc)
            t = Time(dt_utc)
            mjd_val = t.mjd
            self.lbl_out_mjd.setText(f"MJD:\n{mjd_val:.5f}")
        except:
            self.lbl_out_mjd.setText("Error menghitung MJD")

    def calc_date_from_mjd(self):
        try:
            mjd_val = float(self.mjd_input.text().strip())
            t = Time(mjd_val, format='mjd')
            dt_utc = t.datetime.replace(tzinfo=timezone.utc)
            dt_wib = dt_utc + timedelta(hours=7)
            str_utc = dt_utc.strftime('%d %B %Y  %H:%M:%S')
            str_wib = dt_wib.strftime('%d %B %Y  %H:%M:%S')
            self.lbl_out_date.setText(f"Waktu UTC:\n{str_utc}\n\nWaktu WIB:\n{str_wib}")
        except:
            self.lbl_out_date.setText("Error:\nInput MJD tidak valid")

    def setup_remote_tab(self):
        tab_layout = QVBoxLayout(self.tab_remote)
        tab_layout.setContentsMargins(15, 20, 15, 15)

        input_box = QHBoxLayout()
        v_btns = QVBoxLayout()
        v_edits = QVBoxLayout()
        
        v_btns.addWidget(QLabel("No. Order:"), 0, Qt.AlignVCenter)
        v_btns.addWidget(QPushButton("Directory Standard", clicked=lambda: self.get_path(self.txt_std)))
        v_btns.addWidget(QPushButton("Directory UUT", clicked=lambda: self.get_path(self.txt_uut)))

        self.txt_order = QLineEdit(placeholderText="Contoh: E-26-05-100")
        self.txt_std = QLineEdit(readOnly=True)
        self.txt_uut = QLineEdit(readOnly=True)

        v_edits.addWidget(self.txt_order)
        v_edits.addWidget(self.txt_std)
        v_edits.addWidget(self.txt_uut)

        input_box.addLayout(v_btns, 1)
        input_box.addLayout(v_edits, 4)
        tab_layout.addLayout(input_box)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)

        left_side = QVBoxLayout()
        self.btn_start = QPushButton("START ANALYSIS", clicked=self.run_process)
        self.btn_start.setFixedHeight(45)
        self.btn_start.setStyleSheet("background-color: #ff9800; border: 1px solid #ef6c00;")
        
        self.btn_graph = QPushButton("TAMPILKAN GRAFIK", clicked=self.show_graph)
        self.btn_graph.setFixedHeight(35)
        self.btn_graph.setStyleSheet("background-color: #1976d2; border: 1px solid #1565c0;")
        self.btn_graph.setEnabled(False) 

        self.pbar = QProgressBar()
        self.pbar.setFixedHeight(25)
        
        self.lbl_activity = QLabel("Menunggu input...")
        self.lbl_activity.setStyleSheet("color: #d84315; font-style: italic; font-weight: bold;")
        
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("font-family: Consolas; color: #1b5e20; font-weight: bold; margin-top: 10px;")

        left_side.addWidget(self.btn_start)
        left_side.addWidget(self.btn_graph) 
        left_side.addSpacing(10)
        left_side.addWidget(QLabel("Progress:"))
        left_side.addWidget(self.pbar)
        left_side.addWidget(self.lbl_activity)
        left_side.addWidget(self.lbl_summary) 
        left_side.addStretch()

        right_side = QVBoxLayout()
        right_side.addWidget(QLabel("<b>HASIL AKHIR ANALISIS:</b>"))
        self.log = QTextEdit(readOnly=True) 
        self.log.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        self.log.setPlaceholderText("Hasil analisis akan ditampilkan disini...")
        right_side.addWidget(self.log)

        bottom_layout.addLayout(left_side, 1)  
        bottom_layout.addLayout(right_side, 2) 
        
        tab_layout.addLayout(bottom_layout)

    def setup_timer(self):
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)

    def update_time(self):
        utc = datetime.now(timezone.utc)
        wib = utc + timedelta(hours=7)
        mjd = int(Time(utc.date().isoformat(), format='iso').mjd)
        self.lbl_clock.setText(f"MJD: {mjd} | {wib.strftime('%d %b %Y')} | WIB: {wib.strftime('%H:%M:%S')} | UTC: {utc.strftime('%H:%M:%S')}")

    def get_path(self, edit):
        path = QFileDialog.getExistingDirectory(self, "Pilih Folder")
        if path: edit.setText(path)

    def run_check_data(self):
        if not self.txt_std_check.text() or not self.txt_uut_check.text():
            self.lbl_activity_check.setText("Error: Pilih folder terlebih dahulu!")
            return
            
        self.btn_check.setEnabled(False)
        self.log_check.clear()
        self.pbar_check.setValue(0)
        
        self.checker = CheckDataThread(self.txt_std_check.text(), self.txt_uut_check.text())
        self.checker.activity_update.connect(lambda m: self.lbl_activity_check.setText(f"Aktifitas: {m}"))
        self.checker.progress_update.connect(self.pbar_check.setValue)
        self.checker.report_update.connect(self.log_check.setPlainText)
        self.checker.graph_data_update.connect(self.receive_check_graph_data) # <-- Tambahkan ini
        self.checker.finished_update.connect(self.on_check_finish)
        self.checker.error_update.connect(self.on_check_err)
        self.checker.start()

    def on_check_finish(self, msg):
        self.btn_check.setEnabled(True)
        self.lbl_activity_check.setText("Selesai")

    def on_check_err(self, msg):
        self.btn_check.setEnabled(True)
        self.lbl_activity_check.setText("Error terjadi")
        self.log_check.setPlainText(f"[ERROR]\n{msg}")

    def run_process(self):
        if not self.txt_order.text() or not self.txt_std.text() or not self.txt_uut.text():
            self.lbl_activity.setText("Error: Lengkapi input!")
            return
            
        self.btn_start.setEnabled(False)
        self.btn_graph.setEnabled(False) 
        self.lbl_summary.clear() 
        self.log.clear()
        self.pbar.setValue(0)
        
        self.worker = AnalysisThread(self.txt_std.text(), self.txt_uut.text(), self.txt_order.text())
        self.worker.activity_update.connect(lambda m: self.lbl_activity.setText(f"Aktifitas: {m}"))
        self.worker.progress_update.connect(self.pbar.setValue)
        self.worker.finished_update.connect(self.on_finish)
        self.worker.report_update.connect(self.log.setPlainText) 
        self.worker.graph_data_update.connect(self.receive_graph_data) 
        self.worker.error_update.connect(self.on_err)
        self.worker.start()

    def receive_graph_data(self, df):
        self.df_graph = df 

    def on_finish(self, summary):
        self.btn_start.setEnabled(True)
        self.btn_graph.setEnabled(True) 
        self.lbl_summary.setText(summary) 

    def on_err(self, msg):
        self.btn_start.setEnabled(True)
        self.lbl_activity.setText("Proses Berhenti (Error)")
        self.lbl_summary.setText(f"[FATAL ERROR]\n{msg}")

    def show_graph(self):
        if not hasattr(self, 'df_graph') or self.df_graph.empty:
            return
        df = self.df_graph.dropna(subset=['MJD', 'STTIME']).copy()
        if df.empty: return
        x_time = df['MJD'] + df['STTIME'] / 86400.0
        x_pts = np.arange(len(df)) 
        col_std = [c for c in df.columns if 'Terkoreksi' in c][0]
        col_uut = [c for c in df.columns if 'UUT' in c and 'Rata-rata' in c][0]
        col_diff = [c for c in df.columns if 'Selisih' in c][0]
        y_std = df[col_std].astype(float).values
        y_uut = df[col_uut].astype(float).values
        y_diff = df[col_diff].astype(float).values
        m_std, c_std = np.polyfit(x_pts, y_std, 1)
        m_uut, c_uut = np.polyfit(x_pts, y_uut, 1)
        m_diff, c_diff = np.polyfit(x_pts, y_diff, 1)
        eq_std = f"Standar Terkoreksi: y = {m_std:.4f}x + {c_std:.4f}"
        eq_uut = f"UUT: y = {m_uut:.4f}x + {c_uut:.4f}"
        eq_diff = f"Selisih: y = {m_diff:.4f}x + {c_diff:.4f}"
        plt.figure(f"Grafik Analisis Order {self.txt_order.text()}", figsize=(10, 7))
        plt.subplot(2, 1, 1)
        plt.plot(x_time, y_std, 'g.-', label=eq_std, markersize=3)
        plt.plot(x_time, y_uut, 'b.-', label=eq_uut, markersize=3)
        plt.ylabel("Phase / Waktu (ns)")
        plt.title("Perbandingan Standar Terkoreksi dan UUT (Tahap 6)")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.subplot(2, 1, 2)
        plt.plot(x_time, y_diff, 'r.-', label=eq_diff, markersize=3)
        plt.xlabel("MJD (Fraksi)")
        plt.ylabel("Selisih (ns)")
        plt.title("Selisih (Standar - UUT)")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show() 
    def receive_check_graph_data(self, df_std, df_uut):
        self.df_check_std = df_std
        self.df_check_uut = df_uut
        self.btn_check_graph.setEnabled(True)

    def show_check_graph(self):
        plt.figure("Grafik Cek Kelengkapan & Jump Detection (Standard vs UUT)", figsize=(10, 6))

        plt.subplot(2, 1, 1)
        if hasattr(self, 'df_check_std') and not self.df_check_std.empty:
            x_std = self.df_check_std['MJD'] + self.df_check_std['STTIME'] / 86400.0
            y_std = self.df_check_std['REF'].astype(float).values
            plt.plot(x_std, y_std, 'g.-', label='Standard (REFSYS/REFGPS)', markersize=3)
        plt.title("Data Rata-rata Standard")
        plt.ylabel("Waktu (ns)")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 1, 2)
        if hasattr(self, 'df_check_uut') and not self.df_check_uut.empty:
            x_uut = self.df_check_uut['MJD'] + self.df_check_uut['STTIME'] / 86400.0
            y_uut = self.df_check_uut['REF'].astype(float).values
            plt.plot(x_uut, y_uut, 'b.-', label='UUT (REFSYS/REFGPS)', markersize=3)
        plt.title("Data Rata-rata UUT")
        plt.xlabel("MJD (Fraksi)")
        plt.ylabel("Waktu (ns)")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def on_check_finish(self, msg):
        self.btn_check.setEnabled(True)
        self.lbl_activity_check.setText("Selesai")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CGGTTSMainApp()
    ex.show()
    sys.exit(app.exec())