# ==========================================================
# BACA SELURUH SHEET
# ==========================================================

import os
import warnings
import itertools

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


def hitung_metrik(actual, prediksi):
    actual = pd.Series(actual, dtype=float)
    prediksi = pd.Series(prediksi, index=actual.index, dtype=float)

    valid = actual.notna() & prediksi.notna()
    actual = actual[valid]
    prediksi = prediksi[valid]

    mae = mean_absolute_error(actual, prediksi)
    rmse = np.sqrt(mean_squared_error(actual, prediksi))

    mask_mape = actual != 0

    if mask_mape.sum() > 0:
        mape = np.mean(
            np.abs(
                (actual[mask_mape] - prediksi[mask_mape])
                / actual[mask_mape]
            )
        ) * 100
    else:
        mape = np.nan

    if np.abs(actual).sum() > 0:
        wape = (
            np.abs(actual - prediksi).sum()
            / np.abs(actual).sum()
        ) * 100
    else:
        wape = np.nan

    return mae, rmse, mape, wape


file_excel = "DATASET ARIMA SEBELUM PREPROCESSING.xlsm"
excel = pd.ExcelFile(file_excel)
# =====================================================
# FOLDER HASIL GRAFIK
# =====================================================

folder_output = "HASIL_GRAFIK"
os.makedirs(folder_output, exist_ok=True)

print("Jumlah sheet:", len(excel.sheet_names))
print()

for i, sheet in enumerate(excel.sheet_names, start=1):
    print(i, sheet)

print("\nSELESAI")
# =====================================================
# MENYIMPAN HASIL MODEL TERBAIK
# =====================================================
hasil_evaluasi = []
hasil_adf_excel = []
hasil_grid_arima = []
hasil_model_arima = []
hasil_screening = []
# ==========================================================
# LOOP SEMUA PROVINSI
# ==========================================================
total_provinsi = len(excel.sheet_names)

for idx_provinsi, nama_provinsi in enumerate(excel.sheet_names, start=1):

    print("\n")
    print("="*70)
    print(f"PROVINSI {idx_provinsi}/{total_provinsi}")
    print(f"Nama : {nama_provinsi}")
    print("="*70)

    print("\n" + "="*60)
    print("PROVINSI :", nama_provinsi)

    df = pd.read_excel(file_excel, sheet_name=nama_provinsi)
    df = df.dropna(subset=["KABUPATEN/KOTA"])

    print("Jumlah kota :", len(df))
    total_kota = len(df)

# ======================================================
# LOOP SEMUA KABUPATEN/KOTA
# ======================================================

    for i in range(total_kota):

        nama_kota = str(df.iloc[i]["KABUPATEN/KOTA"]).strip()

        # Reset agar hasil kota sebelumnya tidak terbawa
        best = None
        forecast = None
        p = np.nan
        d = np.nan
        q = np.nan

        print("\n")
        print("-"*70)
        print(f"KOTA {i+1}/{total_kota}")
        print(f"Nama : {nama_kota}")
        print("-"*70)

        # ==================================================
        # DATA CURAH HUJAN
        # ==================================================

        # Ambil semua kolom setelah KABUPATEN/KOTA
        kolom_tanggal = df.columns[1:]

        # Konversi nama kolom menjadi tanggal
        tanggal = pd.to_datetime(
            kolom_tanggal,
            errors="coerce"
        )

        # Ambil data curah hujan wilayah tersebut
        data = pd.to_numeric(
            df.iloc[i, 1:].values,
            errors="coerce"
        )

        # Buat time series
        ts = pd.Series(
            data,
            index=tanggal,
            name="curah_hujan"
        )

        # Hapus kolom yang tanggalnya tidak valid
        ts = ts[~ts.index.isna()]
        if ts.isna().any():
            print(f"Ada {ts.isna().sum()} data kosong. Kota dilewati.")
            continue

        if (ts < 0).any():
            print("Ada nilai curah hujan negatif. Kota dilewati.")
            continue

        if ts.nunique() <= 1:
            print("Data konstan. Kota dilewati.")
            continue

        # ==================================================
        # INFORMASI DATA
        # ==================================================

        print("\nINFORMASI DATA")
        print("-" * 40)

        print("Jumlah Data      :", len(ts))
        print("Tanggal Awal     :", ts.index[0].strftime("%Y-%m"))
        print("Tanggal Akhir    :", ts.index[-1].strftime("%Y-%m"))
        print("Missing Value    :", ts.isna().sum())

        # ==================================================
        # TRAIN TEST SPLIT
        # ==================================================

        train = ts.loc["2024-06-01":"2025-11-01"]
        test = ts.loc["2025-12-01":"2026-05-01"]

        print("\nTRAIN TEST SPLIT")
        print("-" * 40)

        print("Jumlah Train :", len(train))
        print("Jumlah Test  :", len(test))

        if len(train) != 18 or len(test) != 6:
            print("Data tidak valid, dilewati.")
            continue
        # ==================================================
        # VALIDATION SPLIT
        # ==================================================

        validation_size = 6

        validation_train = train.iloc[:-validation_size].copy()

        validation_actual = train.iloc[-validation_size:].copy()

        # ==================================================
        # STATISTIK DESKRIPTIF
        # ==================================================

        print("\nSTATISTIK DESKRIPTIF")
        print("-" * 40)

        print(ts.describe())

        # ==================================================
        # CEK DATA
        # ==================================================

        print("\n5 DATA PERTAMA")
        print(ts.head())

        print("\n5 DATA TERAKHIR")
        print(ts.tail())
        # ==================================================
        # ADF TEST
        # ==================================================
        print("\nADF TEST")
        print("-" * 40)

        try:

            hasil_adf = adfuller(train)

            adf_stat = hasil_adf[0]
            p_value = hasil_adf[1]

            print(f"ADF Statistic : {hasil_adf[0]:.6f}")
            print(f"p-value       : {p_value:.6f}")

            print("\nCritical Values")

            for key, value in hasil_adf[4].items():
                print(f"{key} : {value:.6f}")

            if hasil_adf[1] < 0.05:
                status = "Stasioner"
                print("\nKeputusan : Data STASIONER")

            else:
                status = "Tidak Stasioner"
                print("\nKeputusan : Data TIDAK STASIONER")

            hasil_adf_excel.append({
                "Provinsi": nama_provinsi,
                "Kabupaten/Kota": nama_kota,
                "ADF Statistic": adf_stat,
                "p-value": p_value,
                "Critical Value 1%": hasil_adf[4]["1%"],
                "Critical Value 5%": hasil_adf[4]["5%"],
                "Critical Value 10%": hasil_adf[4]["10%"],
                "Status": status
            })
        except Exception as e:
            print("ADF Test gagal:", e)
        # =====================================================
        # MEMBUAT FOLDER PROVINSI
        # =====================================================

        folder_provinsi = os.path.join(
                        folder_output,
                        nama_provinsi.replace(" ", "_")
                    )

        os.makedirs(folder_provinsi, exist_ok=True)

        nama_file = (
                        nama_kota.replace(".", "")
                                .replace("/", "_")
                                .replace(" ", "_")
                    )

        # =====================================================
        # ACF
        # =====================================================

        plt.figure(figsize=(8,4))

        plot_acf(
                        train,
                        lags=min(8, len(train)//2-1)
                    )

        plt.title(f"ACF - {nama_kota}")

        plt.tight_layout()

        plt.savefig(
                        os.path.join(
                            folder_provinsi,
                            f"{nama_file}_ACF.png"
                        )
                    )

        plt.clf()
        plt.close("all")
        # =====================================================
        # PACF
        # =====================================================

        plt.figure(figsize=(8,4))

        plot_pacf(
                        train,
                        lags=min(8, len(train)//2-1),
                        method="ywm"
                    )

        plt.title(f"PACF - {nama_kota}")

        plt.tight_layout()

        plt.savefig(
                        os.path.join(
                            folder_provinsi,
                            f"{nama_file}_PACF.png"
                        )
                    )

        plt.close()

        print("Grafik ACF & PACF berhasil disimpan.")
        # =====================================================
        # GRID SEARCH ARIMA + UJI LJUNG-BOX
        # =====================================================

        print("\n")
        print("="*60)
        print("GRID SEARCH ARIMA")
        print("="*60)

        print("Kriteria pemilihan model:")
        print("1. Model harus lolos uji Ljung-Box (p-value > 0.05)")
        print("2. Model yang lolos diuji menggunakan validation 6 bulan")
        print("3. Model dipilih berdasarkan Validation MAE terkecil")
        print("4. Jika diperlukan, Validation RMSE digunakan sebagai pembanding")
        print("5. AIC digunakan sebagai kriteria berikutnya jika nilai validation sama")
        
        print("-"*60)

        hasil_model = []

        for p, d, q in itertools.product(range(3), range(2), range(3)):

            validation_mae = np.nan
            validation_rmse = np.nan
            validation_mape = np.nan
            validation_wape = np.nan
            status_validation = "Tidak Dicek"
            try:

                # ---------------------------------------------
                # FIT MODEL ARIMA
                # ---------------------------------------------

                model = ARIMA(
                    train,
                    order=(p, d, q)
                )

                model_fit = model.fit()

                # ---------------------------------------------
                # AIC
                # ---------------------------------------------

                aic = model_fit.aic

                # ---------------------------------------------
                # RESIDUAL
                # ---------------------------------------------

                residual = model_fit.resid.dropna()

                # ---------------------------------------------
                # UJI LJUNG-BOX
                # ---------------------------------------------

                # Karena data train hanya 18 observasi,
                # digunakan lag 6 agar tidak terlalu besar.
                lag_lb = min(6, len(residual) - 1)
                model_df = p + q

                # Nilai awal: aman bila uji tidak bisa dilakukan
                lb_stat = np.nan
                lb_pvalue = np.nan
                status_lb = "Derajat bebas tidak cukup"

                if lag_lb > model_df:
                    lb_result = acorr_ljungbox(
                        residual,
                        lags=[lag_lb],
                        model_df=model_df,
                        return_df=True
                    )

                    lb_stat = lb_result["lb_stat"].iloc[0]
                    lb_pvalue = lb_result["lb_pvalue"].iloc[0]

                    if lb_pvalue > 0.05:
                        status_lb = "Lolos"

                        # ---------------------------------------------
                        # VALIDATION 6 BULAN
                        # ---------------------------------------------
                        try:

                            model_validation = ARIMA(
                                validation_train,
                                order=(p, d, q)
                            )

                            fit_validation = model_validation.fit()

                            forecast_validation = fit_validation.forecast(
                                steps=validation_size
                            )

                            forecast_validation = pd.Series(
                                np.maximum(forecast_validation, 0),
                                index=validation_actual.index
                            )

                            (
                                validation_mae,
                                validation_rmse,
                                validation_mape,
                                validation_wape
                            ) = hitung_metrik(
                                validation_actual,
                                forecast_validation
                            )

                            status_validation = "Lolos"

                        except Exception as e:
                            status_validation = "Error"

                            print(
                                f"VALIDATION ERROR | ARIMA({p},{d},{q}) | {e}"
                            )

                    else:
                        status_lb = "Gagal"

                # ---------------------------------------------
                # SIMPAN HASIL
                # ---------------------------------------------

                hasil_model.append([

                    p,
                    d,
                    q,
                    aic,
                    lb_stat,
                    lb_pvalue,
                    status_lb,
                    validation_mae,
                    validation_rmse,
                    validation_mape,
                    validation_wape,
                    status_validation

                ])

                print(
                    f"ARIMA({p},{d},{q})"
                    f" | AIC = {aic:.4f}"
                    f" | LB p-value = {lb_pvalue:.4f}"
                    f" | Val MAE = {validation_mae:.4f}"
                    f" | {status_lb}"
                )

            except Exception as e:

                print(
                    f"ARIMA({p},{d},{q}) gagal : {e}"
                )


        # =====================================================
        # HASIL GRID SEARCH
        # =====================================================

        if len(hasil_model) == 0:
            print("Tidak ada model yang berhasil.")
            best = None

        else:

            hasil_model = pd.DataFrame(

                hasil_model,

                columns=[

                    "p",
                    "d",
                    "q",
                    "AIC",
                    "LB Statistic",
                    "LB p-value",
                    "Status Ljung-Box",
                    "Validation MAE",
                    "Validation RMSE",
                    "Validation MAPE",
                    "Validation WAPE",
                    "Status Validation"
                ]

            )

            # =================================================
            # TAMPILKAN SEMUA MODEL
            # =================================================

            print("\n")
            print("="*70)
            print("HASIL GRID SEARCH ARIMA")
            print("="*70)

            print(hasil_model.to_string(index=False))

            # =================================================
            # FILTER MODEL YANG LOLOS LJUNG-BOX
            # =================================================

            model_lolos = hasil_model[
                (hasil_model["LB p-value"] > 0.05) &
                (hasil_model["Status Validation"] == "Lolos")
            ].copy()

            model_lolos = (
                model_lolos
                .dropna(
                    subset=[
                        "Validation MAE",
                        "Validation RMSE"
                    ]
                )
                .sort_values(
                    by=[
                        "Validation MAE",
                        "Validation RMSE",
                        "AIC"
                    ],
                    ascending=[
                        True,
                        True,
                        True
                    ]
                )
                .reset_index(drop=True)
            )

            print("\n")
            print("="*70)
            print("MODEL YANG LOLOS LJUNG-BOX DAN VALIDATION")
            print("="*70)

            if len(model_lolos) == 0:

                print(
                    "Tidak ada model yang lolos "
                    "Ljung-Box dan validation."
                )

                best = None

            else:

                print(
                    model_lolos.to_string(index=False)
                )

                # =============================================
                # MODEL TERBAIK
                # =============================================

                best = model_lolos.iloc[0]

                print("\n")
                print("="*60)
                print("MODEL ARIMA TERPILIH")
                print("="*60)

                print(
                    f"ARIMA("
                    f"{int(best['p'])},"
                    f"{int(best['d'])},"
                    f"{int(best['q'])})"
                )

                print(
                    f"AIC          = {best['AIC']:.4f}"
                )

                print(
                    f"LB Statistic = {best['LB Statistic']:.4f}"
                )

                print(
                    f"LB p-value   = {best['LB p-value']:.4f}"
                )

                print(
                    f"Status       = {best['Status Ljung-Box']}"
                )

                print(
                    f"Validation MAE = "
                    f"{best['Validation MAE']:.4f}"
                )

                print(
                    f"Validation RMSE = "
                    f"{best['Validation RMSE']:.4f}"
                )

                print(
                    f"Validation MAPE = "
                    f"{best['Validation MAPE']:.2f}%"
                )

                print(
                    f"Validation WAPE = "
                    f"{best['Validation WAPE']:.2f}%"
                )

            # =================================================
            # SIMPAN GRID SEARCH KE EXCEL
            # =================================================

            grid_excel = hasil_model.copy()

            grid_excel.insert(
                0,
                "Provinsi",
                nama_provinsi
            )

            grid_excel.insert(
                1,
                "Kabupaten/Kota",
                nama_kota
            )

            grid_excel["Model"] = (
                "ARIMA("
                + grid_excel["p"].astype(int).astype(str) + ","
                + grid_excel["d"].astype(int).astype(str) + ","
                + grid_excel["q"].astype(int).astype(str) + ")"
            )

            grid_excel = grid_excel[
                [
                    "Provinsi",
                    "Kabupaten/Kota",
                    "Model",
                    "AIC",
                    "LB Statistic",
                    "LB p-value",
                    "Status Ljung-Box",
                    "Validation MAE",
                    "Validation RMSE",
                    "Validation MAPE",
                    "Validation WAPE",
                    "Status Validation"
                ]
            ]

            hasil_grid_arima.append(
                grid_excel
            )
            # =====================================================
            # FORECAST ARIMA
            # =====================================================

            if best is not None:

                print()
                print("="*60)
                print("FORECAST ARIMA")
                print("="*60)

                p = int(best["p"])
                d = int(best["d"])
                q = int(best["q"])

                try:

                    model_final = ARIMA(
                        train,
                        order=(p, d, q)
                    )

                    fit_final = model_final.fit()

                    forecast = fit_final.forecast(
                        steps=len(test)
                    )

                    forecast = pd.Series(
                        np.maximum(forecast, 0),
                        index=test.index
                    )

                    print("\nForecast")
                    print(forecast)

                except Exception as e:

                    print("Forecast gagal :", e)

                    forecast = pd.Series(
                        [np.nan] * len(test),
                        index=test.index
                    )

            else:

                print()
                print("="*60)
                print("TIDAK ADA MODEL ARIMA YANG LOLOS")
                print("="*60)

                forecast = pd.Series(
                    [np.nan] * len(test),
                    index=test.index
                )

                p = np.nan
                d = np.nan
                q = np.nan
        print()
        print("="*60)
        print("EVALUASI")
        print("="*60)

        try:
            mae, rmse, mape, wape = hitung_metrik(
                test,
                forecast
            )
        except Exception as e:
            print("Evaluasi gagal:", e)
            mae = np.nan
            rmse = np.nan
            mape = np.nan
            wape = np.nan
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.2f}%")
        print(f"WAPE : {wape:.2f}%")
        # =====================================================
        # SIMPAN HASIL MODEL ARIMA TERBAIK
        # =====================================================

        if best is not None:

            hasil_model_arima.append({

                "Provinsi":
                    nama_provinsi,

                "Kabupaten/Kota":
                    nama_kota,

                "Model ARIMA":
                    f"ARIMA({p},{d},{q})",

                "p":
                    p,

                "d":
                    d,

                "q":
                    q,

                "AIC":
                    best["AIC"],

                "LB Statistic":
                    best["LB Statistic"],

                "LB p-value":
                    best["LB p-value"],

                "Status Ljung-Box":
                    best["Status Ljung-Box"],

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "MAPE":
                    mape,
                
                "WAPE":
                    wape,
                "Validation MAE":
                    best["Validation MAE"],

                "Validation RMSE":
                    best["Validation RMSE"],

                "Validation MAPE":
                    best["Validation MAPE"],

                "Validation WAPE":
                    best["Validation WAPE"],
            })

        else:

            hasil_model_arima.append({

                "Provinsi":
                    nama_provinsi,

                "Kabupaten/Kota":
                    nama_kota,

                "Model ARIMA":
                    "Tidak ada model lolos",

                "p":
                    np.nan,

                "d":
                    np.nan,

                "q":
                    np.nan,

                "AIC":
                    np.nan,

                "LB Statistic":
                    np.nan,

                "LB p-value":
                    np.nan,

                "Status Ljung-Box":
                    "Tidak ada model lolos",

                "MAE":
                    np.nan,

                "RMSE":
                    np.nan,

                "MAPE":
                    np.nan,

                "WAPE": wape
            })
        if best is not None:
            # ==================================================
            # GRAFIK FORECAST ARIMA + SCREENING KUALITAS GRAFIK
            # ==================================================

            # --------------------------------------------------
            # FOLDER OUTPUT GRAFIK
            # --------------------------------------------------

            folder_forecast = os.path.join(
                folder_output,
                nama_provinsi.replace(" ", "_"),
                "FORECAST"
            )

            os.makedirs(
                folder_forecast,
                exist_ok=True
            )

            # ==================================================
            # SCREENING POLA FORECAST
            # ==================================================

            # --------------------------------------------------
            # Pastikan data aktual dan forecast sejajar
            # --------------------------------------------------

            actual = np.asarray(
                test.values,
                dtype=float
            )

            pred = np.asarray(
                forecast.values,
                dtype=float
            )

            # --------------------------------------------------
            # 1. KORELASI AKTUAL VS FORECAST
            # --------------------------------------------------

            if (
                len(actual) >= 2
                and np.std(actual) > 0
                and np.std(pred) > 0
            ):
                korelasi = np.corrcoef(
                    actual,
                    pred
                )[0, 1]
            else:
                korelasi = 0.0

            # --------------------------------------------------
            # 2. DIRECTION ACCURACY
            # --------------------------------------------------

            # Perubahan aktual
            actual_diff = np.diff(actual)

            # Perubahan forecast
            pred_diff = np.diff(pred)

            # Hilangkan kasus perubahan = 0
            valid_direction = (
                (actual_diff != 0)
                & (pred_diff != 0)
            )

            if np.sum(valid_direction) > 0:

                actual_direction = np.sign(
                    actual_diff[valid_direction]
                )

                pred_direction = np.sign(
                    pred_diff[valid_direction]
                )

                direction_accuracy = (
                    np.mean(
                        actual_direction
                        == pred_direction
                    )
                    * 100
                )

            else:

                direction_accuracy = 0.0

            # --------------------------------------------------
            # 3. CORRELATION SCORE
            # --------------------------------------------------

            # Ubah korelasi -1 sampai 1
            # menjadi skor 0 sampai 100

            correlation_score = (
                (korelasi + 1) / 2
            ) * 100

            # --------------------------------------------------
            # 4. VISUAL SCORE
            # --------------------------------------------------

            # Bobot:
            # 60% kemiripan pola
            # 40% kesamaan arah

            visual_score = (
                0.60 * correlation_score
                +
                0.40 * direction_accuracy
            )

            # ==================================================
            # CETAK HASIL SCREENING
            # ==================================================

            print(
                f"[SCREENING] "
                f"{nama_kota} | "
                f"Korelasi={korelasi:.3f} | "
                f"Direction={direction_accuracy:.2f}% | "
                f"Visual Score={visual_score:.2f}"
            )

            # ==================================================
            # GRAFIK
            # ==================================================

            plt.figure(
                figsize=(12, 6)
            )

            # --------------------------------------------------
            # DATA TRAINING
            # --------------------------------------------------

            plt.plot(
                train.index,
                train.values,
                marker="o",
                label="Data Training"
            )

            # --------------------------------------------------
            # DATA AKTUAL
            # --------------------------------------------------

            plt.plot(
                test.index,
                test.values,
                marker="o",
                label="Data Aktual"
            )

            # --------------------------------------------------
            # FORECAST ARIMA
            # --------------------------------------------------

            plt.plot(
                forecast.index,
                forecast.values,
                marker="o",
                linestyle="--",
                label="Forecast ARIMA"
            )

            # --------------------------------------------------
            # GARIS PEMISAH TRAINING DAN TESTING
            # --------------------------------------------------

            plt.axvline(
                x=test.index[0],
                linestyle=":"
            )

            # --------------------------------------------------
            # JUDUL + METRIK
            # --------------------------------------------------

            plt.title(
                f"Forecast Curah Hujan ARIMA - {nama_kota}"
                f"\nMAE={mae:.2f} | "
                f"RMSE={rmse:.2f} | "
                f"MAPE={mape:.2f}% | "
                f"WAPE={wape:.2f}%"
            )

            # --------------------------------------------------
            # LABEL SUMBU
            # --------------------------------------------------

            plt.xlabel(
                "Waktu"
            )

            plt.ylabel(
                "Curah Hujan"
            )

            # --------------------------------------------------
            # LEGEND
            # --------------------------------------------------

            plt.legend()

            # --------------------------------------------------
            # GRID
            # --------------------------------------------------

            plt.grid(
                True,
                alpha=0.3
            )

            # --------------------------------------------------
            # FORMAT TANGGAL
            # --------------------------------------------------

            plt.gca().xaxis.set_major_formatter(
                mdates.DateFormatter("%b-%Y")
            )

            plt.xticks(
                rotation=45
            )

            # --------------------------------------------------
            # LAYOUT
            # --------------------------------------------------

            plt.tight_layout()

            # --------------------------------------------------
            # NAMA FILE
            # --------------------------------------------------

            nama_file_grafik = (
                f"{nama_provinsi}_{nama_kota}_ARIMA.png"
            )

            nama_file_grafik = (
                nama_file_grafik
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )

            # --------------------------------------------------
            # SIMPAN PNG
            # --------------------------------------------------

            plt.savefig(
                os.path.join(
                    folder_forecast,
                    nama_file_grafik
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            # ==================================================
            # SIMPAN HASIL SCREENING KE LIST
            # ==================================================

            hasil_screening.append({
                "PROVINSI": nama_provinsi,
                "KABUPATEN_KOTA": nama_kota,

                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "WAPE": wape,

                "KORELASI": korelasi,
                "DIRECTION_ACCURACY": direction_accuracy,
                "VISUAL_SCORE": visual_score
            })

            if best is not None:

                hasil_evaluasi.append({

                    "Provinsi":
                        nama_provinsi,

                    "Kota":
                        nama_kota,

                    "Model":
                        f"ARIMA({p},{d},{q})",

                    "MAE":
                        mae,

                    "RMSE":
                        rmse,

                    "MAPE":
                        mape,

                    "WAPE": wape
                })
# ==================================================
# SIMPAN HASIL SCREENING
# ==================================================

df_screening = pd.DataFrame(
    hasil_screening
)

# Urutkan dari Visual Score tertinggi
if not df_screening.empty and "VISUAL_SCORE" in df_screening.columns:

    df_screening = df_screening.sort_values(
        by="VISUAL_SCORE",
        ascending=False
    )

else:

    print("⚠️ HASIL SCREENING GRAFIK KOSONG / KOLOM VISUAL_SCORE TIDAK TERSEDIA")

# --------------------------------------------------
# SIMPAN EXCEL
# --------------------------------------------------

file_screening = os.path.join(
    folder_output,
    "SCREENING_GRAFIK_ARIMA.xlsx"
)

df_screening.to_excel(
    file_screening,
    index=False
)

print("\n" + "=" * 70)
print("TOP 20 GRAFIK ARIMA BERDASARKAN VISUAL SCORE")
print("=" * 70)

if not df_screening.empty:
    print(
        df_screening[
            [
                "PROVINSI",
                "KABUPATEN_KOTA",
                "WAPE",
                "KORELASI",
                "DIRECTION_ACCURACY",
                "VISUAL_SCORE"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
else:
    print("Belum ada grafik yang dapat masuk screening.")

print("\nFile screening berhasil dibuat:")
print(file_screening)
# =====================================================
# SIMPAN KE EXCEL
# =====================================================

# =====================================================
# HASIL EVALUASI
# =====================================================

evaluasi_df = pd.DataFrame(
    hasil_evaluasi
)

evaluasi_df.to_excel(
    "EVALUASI_ARIMA.xlsx",
    index=False
)

# =====================================================
# HASIL ADF
# =====================================================

df_adf = pd.DataFrame(
    hasil_adf_excel
)

df_adf.to_excel(
    "HASIL_ADF_ARIMA.xlsx",
    index=False
)

# =====================================================
# HASIL GRID SEARCH
# =====================================================

if len(hasil_grid_arima) > 0:

    grid_df = pd.concat(
        hasil_grid_arima,
        ignore_index=True
    )

    grid_df.to_excel(
        "GRID_SEARCH_ARIMA.xlsx",
        index=False
    )

    print(
        "GRID_SEARCH_ARIMA.xlsx berhasil disimpan."
    )

else:

    print(
        "GRID_SEARCH_ARIMA.xlsx tidak dibuat "
        "karena tidak ada hasil grid search."
    )

# =====================================================
# HASIL MODEL ARIMA TERBAIK
# =====================================================

df_model_arima = pd.DataFrame(
    hasil_model_arima
)

df_model_arima.to_excel(
    "HASIL_MODEL_ARIMA.xlsx",
    index=False
)

print(
    "HASIL_MODEL_ARIMA.xlsx berhasil disimpan."
)

print("\n")
print("="*60)
print("SEMUA HASIL BERHASIL DISIMPAN")
print("="*60)

print("File : HASIL_ADF_ARIMA.xlsx")
print("File : GRID_SEARCH_ARIMA.xlsx")
print("File : HASIL_MODEL_ARIMA.xlsx")
print("File : EVALUASI_ARIMA.xlsx")
