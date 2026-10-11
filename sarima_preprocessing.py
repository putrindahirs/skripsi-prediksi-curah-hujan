# ==========================================================
# IMPORT
# ==========================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
import warnings
import itertools

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")

# ==========================================================
# FUNGSI HITUNG METRIK EVALUASI
# ==========================================================

def hitung_metrik(actual, prediksi):

    actual = pd.Series(
        actual,
        dtype=float
    )

    prediksi = pd.Series(
        prediksi,
        index=actual.index,
        dtype=float
    )

    # ------------------------------------------------------
    # Ambil data yang valid
    # ------------------------------------------------------

    valid = (
        actual.notna()
        &
        prediksi.notna()
    )

    actual = actual[valid]

    prediksi = prediksi[valid]


    # ------------------------------------------------------
    # MAE
    # ------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        prediksi
    )


    # ------------------------------------------------------
    # RMSE
    # ------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediksi
        )
    )


    # ------------------------------------------------------
    # MAPE
    # ------------------------------------------------------

    mask_mape = (
        actual != 0
    )

    if mask_mape.sum() > 0:

        mape = np.mean(

            np.abs(

                (
                    actual[mask_mape]
                    -
                    prediksi[mask_mape]
                )
                /
                actual[mask_mape]

            )

        ) * 100

    else:

        mape = np.nan


    # ------------------------------------------------------
    # WAPE
    # ------------------------------------------------------

    total_actual = np.abs(
        actual
    ).sum()

    if total_actual > 0:

        wape = (

            np.abs(
                actual - prediksi
            ).sum()

            /

            total_actual

        ) * 100

    else:

        wape = np.nan


    return (
        mae,
        rmse,
        mape,
        wape
    )
# ==========================================================
# KONFIGURASI
# ==========================================================

file_excel = "DATASET SARIMA SETELAH PREPROCESSING.xlsm"

folder_output = "HASIL_SARIMA_PREPROCESSING_FINAL"

folder_forecast = os.path.join(
    folder_output,
    "FORECAST_2D"
)

folder_acf_pacf = os.path.join(
    folder_output,
    "ACF_PACF"
)


os.makedirs(folder_output, exist_ok=True)
os.makedirs(folder_forecast, exist_ok=True)
os.makedirs(folder_acf_pacf, exist_ok=True)


# ==========================================================
# PARAMETER DATA
# ==========================================================

tanggal_awal = "2024-06-01"

jumlah_data = 24

jumlah_train = 18

jumlah_test = 6

periode_musiman = 12


# ==================================================
# PARAMETER MODEL SARIMA
# ==================================================

# Non-seasonal
p_values = range(3)   # 0, 1, 2
d_values = range(2)   # 0, 1
q_values = range(3)   # 0, 1, 2

# Seasonal
P_values = [0, 1]
D_values = [0]
Q_values = [0, 1]

# ==========================================================
# PARAMETER LJUNG-BOX
# ==========================================================

LB_LAG = 6

LB_ALPHA = 0.05


# ==========================================================
# PARAMETER CEK KESTABILAN FORECAST
# ==========================================================
#
# Forecast pada skala log1p akan dibandingkan dengan
# karakteristik data training.
#
# Batas atas:
#
#   mean(train) + 3 * std(train)
#
# tetapi juga minimal harus mencakup nilai maksimum training.
#
# Jadi:
#
#   batas_atas =
#   max(max_train, mean_train + 3*std_train)
#
# Tujuannya untuk mendeteksi forecast yang melonjak
# sangat jauh dari pola data training.
#
# Contoh Tapanuli Tengah:
#
# train log1p sekitar 0 - 3.5
#
# forecast:
# 5 -> 8 -> 16 -> 35 -> 79 -> 181
#
# akan dianggap tidak stabil.
#
# ==========================================================

STABILITY_SIGMA = 3


# ==========================================================
# FUNGSI CEK FORECAST STABIL
# ==========================================================

def cek_kestabilan_forecast(
    forecast_preprocessed,
    train_preprocessed
):

    forecast_preprocessed = np.asarray(
        forecast_preprocessed,
        dtype=float
    )

    train_preprocessed = np.asarray(
        train_preprocessed,
        dtype=float
    )

    # ------------------------------------------------------
    # CEK 1
    # Apakah ada NaN / infinite?
    # ------------------------------------------------------

    if not np.all(np.isfinite(forecast_preprocessed)):

        return (
            False,
            "Forecast mengandung NaN/Inf",
            np.nan
        )


    # ------------------------------------------------------
    # CEK 2
    # Karena data asli rainfall >= 0,
    # maka pada skala log1p juga seharusnya >= 0.
    # ------------------------------------------------------

    if np.any(forecast_preprocessed < 0):

        return (
            False,
            "Forecast log1p menghasilkan nilai < 0",
            np.nan
        )


    # ------------------------------------------------------
    # Statistik training
    # ------------------------------------------------------

    mean_train = np.mean(train_preprocessed)

    std_train = np.std(
        train_preprocessed,
        ddof=1
    )

    max_train = np.max(train_preprocessed)


    # ------------------------------------------------------
    # Jika std tidak valid / 0
    # ------------------------------------------------------

    if not np.isfinite(std_train):

        return (
            False,
            "Standar deviasi training tidak valid",
            np.nan
        )


    # ------------------------------------------------------
    # Batas kestabilan
    # ------------------------------------------------------

    batas_atas = max(
        max_train,
        mean_train + STABILITY_SIGMA * std_train
    )


    # ------------------------------------------------------
    # Forecast maksimum
    # ------------------------------------------------------

    max_forecast = np.max(
        forecast_preprocessed
    )


    # ------------------------------------------------------
    # CEK
    # ------------------------------------------------------

    if max_forecast > batas_atas:

        return (
            False,
            "Forecast melebihi batas kestabilan",
            batas_atas
        )


    # ------------------------------------------------------
    # Jika lolos
    # ------------------------------------------------------

    return (
        True,
        "Stabil",
        batas_atas
    )


# ==========================================================
# BUKA FILE EXCEL
# ==========================================================

print("=" * 70)
print("MEMBUKA FILE EXCEL")
print("=" * 70)

excel = pd.ExcelFile(file_excel)

print(
    "Jumlah sheet:",
    len(excel.sheet_names)
)

print(
    "Sheet:",
    excel.sheet_names
)


# ==========================================================
# LIST HASIL
# ==========================================================

hasil_evaluasi = []

hasil_adf = []

hasil_model = []

hasil_grid = []

hasil_screening = []
# ==========================================================
# LOOP SELURUH SHEET
# ==========================================================

for nama_sheet in excel.sheet_names:

    print("\n")
    print("=" * 70)
    print("SHEET:", nama_sheet)
    print("=" * 70)


    try:

        df = pd.read_excel(
            file_excel,
            sheet_name=nama_sheet
        )


    except Exception as e:

        print(
            "Gagal membaca sheet:",
            nama_sheet
        )

        print(e)

        continue


    # ======================================================
    # CEK KOLOM
    # ======================================================

    if "KABUPATEN/KOTA" not in df.columns:

        print(
            "Kolom KABUPATEN/KOTA tidak ditemukan."
        )

        continue


    # ======================================================
    # LOOP KABUPATEN/KOTA
    # ======================================================

    for i in range(len(df)):

        kota = df.iloc[i]["KABUPATEN/KOTA"]


        # --------------------------------------------------
        # Skip jika nama kota kosong
        # --------------------------------------------------

        if pd.isna(kota):

            continue


        kota = str(kota).strip()


        print("\n")
        print("-" * 70)
        print(
            "PROVINSI:",
            nama_sheet
        )
        print(
            "KABUPATEN/KOTA:",
            kota
        )
        print("-" * 70)


        # ==================================================
        # AMBIL DATA CURAH HUJAN
        # ==================================================

        try:

            # Ambil seluruh kolom setelah KABUPATEN/KOTA
            kolom_tanggal = df.columns[1:]

            # Ubah nama kolom menjadi tanggal
            tanggal = pd.to_datetime(
                kolom_tanggal,
                errors="coerce"
            )

            # Ambil nilai curah hujan
            data = pd.to_numeric(
                df.iloc[i, 1:].values,
                errors="coerce"
            )

        except Exception as e:

            print(
                "Data rainfall tidak dapat dikonversi."
            )

            print(e)

            continue


        # ==================================================
        # TIME SERIES
        # ==================================================

        ts = pd.Series(
            data,
            index=tanggal,
            dtype=float,
            name="curah_hujan"
        )

        # Hapus kolom yang tanggalnya tidak valid
        ts = ts[~ts.index.isna()]


        # ==================================================
        # CEK JUMLAH DATA
        # ==================================================

        if len(ts) != jumlah_data:

            print(
                "Jumlah data bukan 24. Skip."
            )

            continue


        # ==================================================
        # CEK DATA NEGATIF
        # ==================================================

        if (ts < 0).any():

            print(
                "Terdapat rainfall negatif. Skip."
            )

            continue


        # ==================================================
        # MISSING VALUE SEBELUM
        # ==================================================

        missing_sebelum = int(
            ts.isna().sum()
        )


        # ==================================================
        # HANDLE MISSING VALUE
        # ==================================================

        if missing_sebelum > 0:

            print(
                "Missing value:",
                missing_sebelum
            )

            ts = ts.interpolate(
                method="linear"
            )

            ts = ts.bfill()

            ts = ts.ffill()


        # ==================================================
        # MISSING VALUE SESUDAH
        # ==================================================

        missing_sesudah = int(
            ts.isna().sum()
        )


        # ==================================================
        # JIKA MASIH MISSING
        # ==================================================

        if missing_sesudah > 0:

            print(
                "Masih terdapat missing value. Skip."
            )

            continue


        # ==================================================
        # SPLIT DATA
        # ==================================================

        train = ts.loc[
            "2024-06-01":"2025-11-01"
        ].copy()

        test = ts.loc[
            "2025-12-01":"2026-05-01"
        ].copy()


        # ==================================================
        # PREPROCESSING LOG1P
        # ==================================================

        ts_preprocessed = np.log1p(ts)


        train_preprocessed = ts_preprocessed.loc[
            "2024-06-01":"2025-11-01"
        ].copy()

        # ==================================================
        # VALIDATION DATA
        # ==================================================

        validation_size = 6

        validation_train_preprocessed = (
            train_preprocessed.iloc[:-validation_size]
            .copy()
        )

        validation_actual = (
            train.iloc[-validation_size:]
            .copy()
        )
        # ==================================================
        # ADF TEST
        # ==================================================

        try:

            adf_result = adfuller(
                train_preprocessed.dropna(),
                autolag="AIC"
            )

            adf_statistic = adf_result[0]

            adf_pvalue = adf_result[1]

            adf_lags = adf_result[2]

            adf_nobs = adf_result[3]

            if adf_pvalue < 0.05:

                adf_status = "Stasioner"

            else:

                adf_status = "Tidak Stasioner"


        except Exception as e:

            print(
                "ADF gagal:",
                e
            )

            adf_statistic = np.nan

            adf_pvalue = np.nan

            adf_lags = np.nan

            adf_nobs = np.nan

            adf_status = "ADF Gagal"


        # ==================================================
        # SIMPAN HASIL ADF
        # ==================================================

        hasil_adf.append({

            "Provinsi": nama_sheet,
            "Kota": kota,
            "ADF Statistic": adf_statistic,
            "ADF p-value": adf_pvalue,
            "ADF Lags": adf_lags,
            "ADF Nobs": adf_nobs,
            "Status ADF": adf_status

        })


        # ==================================================
        # ACF & PACF
        # ==================================================

        try:

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            plot_acf(
                train_preprocessed,
                lags=min(
                    8,
                    len(train_preprocessed) // 2
                ),
                ax=ax
            )

            ax.set_title(
                f"ACF SARIMA - {kota}"
            )

            plt.tight_layout()

            nama_file_acf = (
                f"{nama_sheet}_{kota}_ACF.png"
            )

            nama_file_acf = (
                nama_file_acf
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )

            plt.savefig(
                os.path.join(
                    folder_acf_pacf,
                    nama_file_acf
                ),
                dpi=150
            )

            plt.close()


        except Exception:

            pass


        try:

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            plot_pacf(
                train_preprocessed,
                lags=min(
                    8,
                    len(train_preprocessed) // 2
                ),
                ax=ax,
                method="ywm"
            )

            ax.set_title(
                f"PACF SARIMA - {kota}"
            )

            plt.tight_layout()

            nama_file_pacf = (
                f"{nama_sheet}_{kota}_PACF.png"
            )

            nama_file_pacf = (
                nama_file_pacf
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )

            plt.savefig(
                os.path.join(
                    folder_acf_pacf,
                    nama_file_pacf
                ),
                dpi=150
            )

            plt.close()


        except Exception:

            pass


        # ==================================================
        # GRID SEARCH SARIMA
        # ==================================================

        hasil_model_kota = []


        semua_model = list(
            itertools.product(
                p_values,
                d_values,
                q_values,
                P_values,
                D_values,
                Q_values
            )
        )


        print(
            "Jumlah kandidat model:",
            len(semua_model)
        )


        # ==================================================
        # LOOP MODEL
        # ==================================================

        for (
            p,
            d,
            q,
            P,
            D,
            Q
        ) in semua_model:


            model_name = (
                f"SARIMA({p},{d},{q})"
                f"({P},{D},{Q},{periode_musiman})"
            )

            # ------------------------------------------
            # HASIL VALIDATION
            # ------------------------------------------

            validation_mae = np.nan
            validation_rmse = np.nan
            validation_mape = np.nan
            validation_wape = np.nan
            status_validation = "Tidak Dicek"


            try:

                # ------------------------------------------
                # FIT MODEL
                # ------------------------------------------

                model = SARIMAX(

                    train_preprocessed,

                    order=(
                        p,
                        d,
                        q
                    ),

                    seasonal_order=(
                        P,
                        D,
                        Q,
                        periode_musiman
                    ),

                    enforce_stationarity=True,

                    enforce_invertibility=True

                )


                model_fit = model.fit(
                    disp=False,
                    maxiter=200
                )


                # ------------------------------------------
                # AIC
                # ------------------------------------------

                aic = model_fit.aic


                # ------------------------------------------
                # RESIDUAL
                # ------------------------------------------

                residual = (
                    model_fit.resid
                    .dropna()
                )


                # ------------------------------------------
                # LJUNG-BOX
                # ------------------------------------------

                model_df = (
                    p
                    + q
                    + P
                    + Q
                )

                lag_lb = min(
                    LB_LAG,
                    len(residual) - 1
                )

                lb_statistic = np.nan
                lb_pvalue = np.nan


                if lag_lb > model_df:

                    try:

                        lb_result = acorr_ljungbox(

                            residual,

                            lags=[lag_lb],

                            model_df=model_df,

                            return_df=True

                        )

                        lb_statistic = (
                            lb_result[
                                "lb_stat"
                            ].iloc[0]
                        )

                        lb_pvalue = (
                            lb_result[
                                "lb_pvalue"
                            ].iloc[0]
                        )

                    except Exception:

                        lb_statistic = np.nan

                        lb_pvalue = np.nan

                # ------------------------------------------
                # STATUS LJUNG-BOX
                # ------------------------------------------

                if (
                    np.isfinite(lb_pvalue)
                    and
                    lb_pvalue > LB_ALPHA
                ):

                    status_lb = "Lolos"

                else:

                    status_lb = "Tidak Lolos"


                # ------------------------------------------
                # FORECAST UNTUK SCREENING
                #
                # HANYA dilakukan jika LB lolos.
                # ------------------------------------------

                forecast_preprocessed = np.nan

                status_stability = (
                    "Tidak Dicek"
                )

                alasan_stability = ""

                batas_stability = np.nan

                max_forecast_log = np.nan


                if status_lb == "Lolos":

                    forecast_preprocessed = (
                        model_fit.forecast(
                            steps=jumlah_test
                        )
                    )


                    (
                        stabil,
                        alasan_stability,
                        batas_stability
                    ) = cek_kestabilan_forecast(

                        forecast_preprocessed,

                        train_preprocessed

                    )


                    max_forecast_log = np.max(
                        np.asarray(
                            forecast_preprocessed,
                            dtype=float
                        )
                    )


                    if stabil:

                        status_stability = (
                            "Stabil"
                        )

                    else:

                        status_stability = (
                            "Tidak Stabil"
                        )

                # ------------------------------------------
                # VALIDATION FORECAST
                # ------------------------------------------

                if (
                    status_lb == "Lolos"
                    and status_stability == "Stabil"
                ):

                    try:

                        # Fit ulang kandidat pada data
                        # training validation
                        model_validation = SARIMAX(

                            validation_train_preprocessed,

                            order=(
                                p,
                                d,
                                q
                            ),

                            seasonal_order=(
                                P,
                                D,
                                Q,
                                periode_musiman
                            ),

                            enforce_stationarity=True,

                            enforce_invertibility=True

                        )

                        fit_validation = model_validation.fit(
                            disp=False,
                            maxiter=200
                        )

                        # Forecast 6 bulan validation
                        forecast_validation_preprocessed = (
                            fit_validation.forecast(
                                steps=validation_size
                            )
                        )

                        # Kembalikan dari log1p
                        forecast_validation = pd.Series(

                            np.expm1(
                                forecast_validation_preprocessed
                            ),

                            index=validation_actual.index

                        )

                        # Hindari nilai negatif
                        forecast_validation = forecast_validation.clip(
                            lower=0
                        )

                        # Hitung error validation
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

                else:

                    status_validation = "Tidak Dicek"
                # ------------------------------------------
                # STATUS AKHIR MODEL
                # ------------------------------------------

                if (
                    status_lb == "Lolos"
                    and
                    status_stability == "Stabil"
                    and
                    status_validation == "Lolos"
                ):

                    status_model = "Lolos"

                elif status_lb != "Lolos":

                    status_model = (
                        "Tidak Lolos - Ljung-Box"
                    )

                elif status_stability != "Stabil":

                    status_model = (
                        "Tidak Lolos - Stabilitas"
                    )

                else:

                    status_model = (
                        "Tidak Lolos - Validation"
                    )
                # ------------------------------------------
                # SIMPAN GRID
                # ------------------------------------------

                hasil_model_kota.append({

                    "Provinsi": nama_sheet,

                    "Kota": kota,

                    "Model": model_name,

                    # ==========================================
                    # PARAMETER SARIMA
                    # ==========================================

                    "p": p,
                    "d": d,
                    "q": q,

                    "P": P,
                    "D": D,
                    "Q": Q,

                    "m": periode_musiman,
                    "Model DF": model_df,

                    # ==========================================
                    # HASIL MODEL
                    # ==========================================

                    "AIC": aic,

                    "LB Statistic": lb_statistic,

                    "LB p-value": lb_pvalue,

                    "Status LB": status_lb,

                    "Max Forecast log1p":
                        max_forecast_log,

                    "Batas Stabilitas log1p":
                        batas_stability,

                    "Status Stabilitas":
                        status_stability,

                    "Alasan Stabilitas":
                        alasan_stability,

                    "Status Model":
                        status_model,

                    "Validation MAE": validation_mae,

                    "Validation RMSE": validation_rmse,

                    "Validation MAPE": validation_mape,

                    "Validation WAPE": validation_wape,

                    "Status Validation": status_validation,

                })


                # ------------------------------------------
                # HASIL GRID GLOBAL
                # ------------------------------------------

                hasil_grid.append({

                    "Provinsi": nama_sheet,

                    "Kota": kota,

                    "Model": model_name,

                    "Model DF": model_df,

                    "Validation MAE":
                        validation_mae,

                    "Validation RMSE":
                        validation_rmse,

                    "Validation MAPE":
                        validation_mape,

                    "Validation WAPE":
                        validation_wape,

                    "Status Validation":
                        status_validation,

                    # ==========================================
                    # PARAMETER SARIMA
                    # ==========================================

                    "p": p,
                    "d": d,
                    "q": q,

                    "P": P,
                    "D": D,
                    "Q": Q,

                    "m": periode_musiman,

                    # ==========================================
                    # HASIL MODEL
                    # ==========================================

                    "AIC": aic,

                    "LB Statistic": lb_statistic,

                    "LB p-value": lb_pvalue,

                    "Status LB": status_lb,

                    "Max Forecast log1p":
                        max_forecast_log,

                    "Batas Stabilitas log1p":
                        batas_stability,

                    "Status Stabilitas":
                        status_stability,

                    "Alasan Stabilitas":
                        alasan_stability,

                    "Status Model":
                        status_model

                })


                print(
                    model_name,
                    "| AIC =",
                    round(aic, 3),
                    "| LB p =",
                    round(lb_pvalue, 4)
                    if np.isfinite(lb_pvalue)
                    else "NaN",
                    "| LB =",
                    status_lb,
                    "| Stabilitas =",
                    status_stability
                )


            except Exception as e:

                    # ------------------------------------------
                    # MODEL GAGAL
                    # ------------------------------------------

                    hasil_grid.append({

                        "Provinsi": nama_sheet,

                        "Kota": kota,

                        "Model": model_name,

                        "Model DF":
                            p + q + P + Q,

                        # ==========================================
                        # HASIL VALIDATION
                        # ==========================================

                        "Validation MAE":
                            np.nan,

                        "Validation RMSE":
                            np.nan,

                        "Validation MAPE":
                            np.nan,

                        "Validation WAPE":
                            np.nan,

                        "Status Validation":
                            "Error",

                        # ==========================================
                        # HASIL MODEL
                        # ==========================================

                        "AIC":
                            np.nan,

                        "LB Statistic":
                            np.nan,

                        "LB p-value":
                            np.nan,

                        "Status LB":
                            "Error",

                        "Max Forecast log1p":
                            np.nan,

                        "Batas Stabilitas log1p":
                            np.nan,

                        "Status Stabilitas":
                            "Error",

                        "Alasan Stabilitas":
                            str(e),

                        "Status Model":
                            "Error"

                    })

                    print(
                        model_name,
                        "| ERROR"
                    )


        # ==================================================
        # DATAFRAME HASIL MODEL KOTA
        # ==================================================

        df_model_kota = pd.DataFrame(
            hasil_model_kota
        )


        # ==================================================
        # PILIH MODEL YANG LOLOS
        # ==================================================

        if df_model_kota.empty:

            print(
                "Tidak ada model yang berhasil."
            )

            continue


        model_lolos = df_model_kota[
            df_model_kota[
                "Status Model"
            ] == "Lolos"
        ].copy()


        # ==================================================
        # CEK MODEL LOLOS
        # ==================================================

        if model_lolos.empty:

            print(
                "Tidak ada model yang lolos "
                "Ljung-Box + stabilitas."
            )

            # ----------------------------------------------
            # SIMPAN BARIS TANPA HASIL FORECAST
            # ----------------------------------------------

            hasil_evaluasi.append({

                "Provinsi":
                    nama_sheet,

                "Kota":
                    kota,

                "Model":
                    "Tidak ada model lolos",

                "Preprocessing":
                    "log1p",

                "MAE":
                    np.nan,

                "RMSE":
                    np.nan,

                "MAPE":
                    np.nan,

                "WAPE":
                    np.nan,   

                "AIC":
                    np.nan,

                "LB Statistic":
                    np.nan,

                "LB p-value":
                    np.nan,

                "Status LB":
                    "Tidak ada yang lolos",

                "Status Stabilitas":
                    "Tidak ada yang lolos"

            })

            continue


        # ==================================================
        # PEMILIHAN MODEL BERDASARKAN VALIDATION ERROR
        # ==================================================

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

        best = model_lolos.iloc[0]


        best_model_name = best[
            "Model"
        ]

        best_aic = best[
            "AIC"
        ]

        best_lb_statistic = best[
            "LB Statistic"
        ]

        best_lb_pvalue = best[
            "LB p-value"
        ]


        print("\n")
        print("=" * 70)
        print("MODEL TERPILIH")
        print("=" * 70)

        print(
            "Kota:",
            kota
        )

        print(
            "Model:",
            best_model_name
        )

        print(
            "AIC:",
            best_aic
        )

        print(
            "Validation MAE:",
            best["Validation MAE"]
        )

        print(
            "Validation RMSE:",
            best["Validation RMSE"]
        )

        print(
            "Validation MAPE:",
            best["Validation MAPE"]
        )

        print(
            "Validation WAPE:",
            best["Validation WAPE"]
        )

        print(
            "LB Statistic:",
            best_lb_statistic
        )

        print(
            "LB p-value:",
            best_lb_pvalue
        )

        print(
            "Status LB:",
            best["Status LB"]
        )

        print(
            "Status Stabilitas:",
            best["Status Stabilitas"]
        )


        # ==================================================
        # AMBIL PARAMETER MODEL TERPILIH
        # ==================================================

        # Parameter model sudah disimpan langsung
        # ketika proses grid search.
        #
        # Jadi TIDAK perlu lagi membaca parameter
        # dari nama string "SARIMA(...)".

        p = int(best["p"])
        d = int(best["d"])
        q = int(best["q"])

        P = int(best["P"])
        D = int(best["D"])
        Q = int(best["Q"])

        periode_musiman = int(best["m"])


        print(
            "Parameter model:",
            f"SARIMA({p},{d},{q})"
            f"({P},{D},{Q},{periode_musiman})"
        )


        # ==================================================
        # FIT ULANG MODEL FINAL
        # ==================================================

        model_final = SARIMAX(

            train_preprocessed,

            order=(
                p,
                d,
                q
            ),

            seasonal_order=(
                P,
                D,
                Q,
                periode_musiman
            ),

            enforce_stationarity=True,

            enforce_invertibility=True

        )


        fit_final = model_final.fit(
            disp=False,
            maxiter=200
        )


        # ==================================================
        # FORECAST SKALA LOG1P
        # ==================================================

        forecast_preprocessed = (
            fit_final.forecast(
                steps=jumlah_test
            )
        )


        # ==================================================
        # INVERSE TRANSFORM
        # ==================================================

        forecast = pd.Series(

            np.expm1(
                forecast_preprocessed
            ),

            index=test.index

        )


        # ==================================================
        # CLIP NEGATIF
        # ==================================================

        forecast = forecast.clip(
            lower=0
        )


        # ==================================================
        # CEK FORECAST FINAL
        # ==================================================

        if not np.all(
            np.isfinite(
                forecast.values
            )
        ):

            print(
                "Forecast final tidak valid."
            )

            continue


        # ==================================================
        # EVALUASI MAE, RMSE, MAPE, WAPE
        # ==================================================

        try:

            (
                mae,
                rmse,
                mape,
                wape
            ) = hitung_metrik(
                test,
                forecast
            )

        except Exception as e:

            print(
                "Evaluasi gagal:",
                e
            )

            mae = np.nan

            rmse = np.nan

            mape = np.nan

            wape = np.nan

        # ==================================================
        # CETAK HASIL
        # ==================================================

        print("\n")
        print("=" * 70)
        print("HASIL EVALUASI")
        print("=" * 70)

        print(
            "MAE  :",
            mae
        )

        print(
            "RMSE :",
            rmse
        )

        print(
            "MAPE :",
            mape,
            "%"
        )

        print(
            "WAPE :",
            wape,
            "%"
        )


        # ==================================================
        # SIMPAN EVALUASI
        # ==================================================

        hasil_evaluasi.append({

            "Provinsi":
                nama_sheet,

            "Kota":
                kota,

            "Model":
                best_model_name,

            "Preprocessing":
                "log1p",

            "MAE":
                mae,

            "RMSE":
                rmse,

            "MAPE":
                mape,

            "WAPE":
                wape,

            "AIC":
                best_aic,

            "LB Statistic":
                best_lb_statistic,

            "LB p-value":
                best_lb_pvalue,

            "Status LB":
                best["Status LB"],

            "Status Stabilitas":
                best["Status Stabilitas"],

            "Max Forecast log1p":
                best["Max Forecast log1p"],

            "Batas Stabilitas log1p":
                best["Batas Stabilitas log1p"]

        })


        # ==================================================
        # SIMPAN MODEL TERPILIH
        # ==================================================

        hasil_model.append({

            "Provinsi":
                nama_sheet,

            "Kota":
                kota,

            "Model":
                best_model_name,

            "p":
                p,

            "d":
                d,

            "q":
                q,

            "P":
                P,

            "D":
                D,

            "Q":
                Q,

            "m":
                periode_musiman,

            "AIC":
                best_aic,

})

        # ==================================================
        # GRAFIK FORECAST SARIMA DENGAN PREPROCESSING
        # + SCREENING VISUAL
        # ==================================================

        # --------------------------------------------------
        # DATA AKTUAL DAN FORECAST
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
        # KORELASI AKTUAL VS FORECAST
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
        # DIRECTION ACCURACY
        # --------------------------------------------------

        actual_diff = np.diff(actual)
        pred_diff = np.diff(pred)

        mask = (
            (actual_diff != 0)
            &
            (pred_diff != 0)
        )

        if np.sum(mask) > 0:

            direction_accuracy = (
                np.mean(
                    np.sign(actual_diff[mask])
                    ==
                    np.sign(pred_diff[mask])
                )
                * 100
            )

        else:

            direction_accuracy = 0.0


        # --------------------------------------------------
        # KORELASI KE SKALA 0-100
        # --------------------------------------------------

        correlation_score = (
            (korelasi + 1) / 2
        ) * 100


        # --------------------------------------------------
        # VISUAL SCORE
        # --------------------------------------------------
        # 60% kesesuaian pola
        # 40% kesesuaian arah perubahan
        # --------------------------------------------------

        visual_score = (
            0.60 * correlation_score
            +
            0.40 * direction_accuracy
        )


        # --------------------------------------------------
        # PRINT SCREENING
        # --------------------------------------------------

        print(
            f"[SCREENING SARIMA PREPROCESSING] "
            f"{kota} | "
            f"WAPE={wape:.2f}% | "
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
        # FORECAST SARIMA
        # --------------------------------------------------

        plt.plot(
            forecast.index,
            forecast.values,
            marker="o",
            linestyle="--",
            label="Forecast SARIMA"
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
            f"Forecast Curah Hujan SARIMA Dengan Preprocessing - {kota}"
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
            f"{nama_sheet}_{kota}_SARIMA_PREPROCESSING.png"
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
        # SIMPAN HASIL SCREENING
        # ==================================================

        hasil_screening.append({

            "PROVINSI":
                nama_sheet,

            "KABUPATEN_KOTA":
                kota,

            "MODEL":
                best_model_name,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "MAPE":
                mape,

            "WAPE":
                wape,

            "KORELASI":
                korelasi,

            "DIRECTION_ACCURACY":
                direction_accuracy,

            "VISUAL_SCORE":
                visual_score

        })


        # ==================================================
        # PRINT FORECAST
        # ==================================================

        print("\nForecast:")

        for tanggal_forecast, nilai in forecast.items():

            print(
                tanggal_forecast.strftime("%Y-%m"),
                ":",
                nilai
            )

# ==========================================================
# HASIL SCREENING GRAFIK SARIMA PREPROCESSING
# ==========================================================

if len(hasil_screening) > 0:

    df_screening = pd.DataFrame(
        hasil_screening
    )

    # Urutkan Visual Score tertinggi
    df_screening = df_screening.sort_values(
        by="VISUAL_SCORE",
        ascending=False
    )

    # ------------------------------------------------------
    # SIMPAN EXCEL
    # ------------------------------------------------------

    file_screening = os.path.join(
        folder_output,
        "SCREENING_GRAFIK_SARIMA_PREPROCESSING.xlsx"
    )

    df_screening.to_excel(
        file_screening,
        index=False
    )

    # ------------------------------------------------------
    # TAMPILKAN TOP 20
    # ------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "TOP 20 SCREENING GRAFIK "
        "SARIMA PREPROCESSING"
    )
    print("=" * 80)

    print(
        df_screening[
            [
                "PROVINSI",
                "KABUPATEN_KOTA",
                "MODEL",
                "WAPE",
                "KORELASI",
                "DIRECTION_ACCURACY",
                "VISUAL_SCORE"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n")
    print(
        "File screening berhasil disimpan:"
    )

    print(
        file_screening
    )

else:

    print(
        "\nTidak ada data screening "
        "yang berhasil dikumpulkan."
    )
# ==========================================================
# DATAFRAME HASIL AKHIR
# ==========================================================

df_evaluasi = pd.DataFrame(
    hasil_evaluasi
)

df_adf = pd.DataFrame(
    hasil_adf
)

df_model = pd.DataFrame(
    hasil_model
)

df_grid = pd.DataFrame(
    hasil_grid
)


# ==========================================================
# SIMPAN KE EXCEL
# ==========================================================

file_evaluasi = os.path.join(

    folder_output,

    "EVALUASI_SARIMA_PREPROCESSING_FINAL.xlsx"

)


file_adf = os.path.join(

    folder_output,

    "HASIL_ADF_SARIMA_PREPROCESSING_FINAL.xlsx"

)


file_model = os.path.join(

    folder_output,

    "HASIL_MODEL_SARIMA_PREPROCESSING_FINAL.xlsx"

)


file_grid = os.path.join(

    folder_output,

    "GRID_SEARCH_SARIMA_PREPROCESSING_FINAL.xlsx"

)


# ==========================================================
# EXCEL EVALUASI
# ==========================================================

df_evaluasi.to_excel(
    file_evaluasi,
    index=False
)


# ==========================================================
# EXCEL ADF
# ==========================================================

df_adf.to_excel(
    file_adf,
    index=False
)


# ==========================================================
# EXCEL MODEL TERPILIH
# ==========================================================

df_model.to_excel(
    file_model,
    index=False
)


# ==========================================================
# EXCEL GRID SEARCH
# ==========================================================

df_grid.to_excel(
    file_grid,
    index=False
)


# ==========================================================
# LAPORAN LENGKAP
# ==========================================================

file_laporan = os.path.join(

    folder_output,

    "LAPORAN_LENGKAP_SARIMA_PREPROCESSING_FINAL.xlsx"

)


with pd.ExcelWriter(
    file_laporan,
    engine="openpyxl"
) as writer:

    df_evaluasi.to_excel(
        writer,
        sheet_name="Evaluasi",
        index=False
    )

    df_adf.to_excel(
        writer,
        sheet_name="ADF",
        index=False
    )

    df_model.to_excel(
        writer,
        sheet_name="Model Terpilih",
        index=False
    )

    df_grid.to_excel(
        writer,
        sheet_name="Grid Search",
        index=False
    )


# ==========================================================
# RINGKASAN
# ==========================================================

print("\n")
print("=" * 70)
print("SELESAI")
print("=" * 70)


print(
    "Jumlah Kabupaten/Kota berhasil:",
    len(df_evaluasi)
)


print(
    "Folder output:",
    folder_output
)


print("\nFile yang dibuat:")

print(
    "1.",
    file_evaluasi
)

print(
    "2.",
    file_adf
)

print(
    "3.",
    file_model
)

print(
    "4.",
    file_grid
)

print(
    "5.",
    file_laporan
)

print(
    "6. Folder grafik:",
    folder_forecast
)

print(
    "7. Folder ACF/PACF:",
    folder_acf_pacf
)


# ==========================================================
# RINGKASAN JUMLAH STATUS
# ==========================================================

if not df_grid.empty:

    print("\n")
    print("=" * 70)
    print("RINGKASAN GRID SEARCH")
    print("=" * 70)

    print(
        "Total kandidat:",
        len(df_grid)
    )

    print(
        "Lolos:",
        (
            df_grid["Status Model"]
            == "Lolos"
        ).sum()
    )

    print(
        "Tidak lolos Ljung-Box:",
        df_grid[
            "Status Model"
        ].astype(str).str.contains(
            "Ljung-Box"
        ).sum()
    )

    print(
        "Tidak lolos stabilitas:",
        df_grid[
            "Status Model"
        ].astype(str).str.contains(
            "Stabilitas"
        ).sum()
    )

    print(
        "Error:",
        (
            df_grid["Status Model"]
            == "Error"
        ).sum()
    )


# ==========================================================
# RINGKASAN EVALUASI
# ==========================================================

if not df_evaluasi.empty:

    print("\n")
    print("=" * 70)
    print("RINGKASAN EVALUASI")
    print("=" * 70)

    print(
        "Jumlah hasil evaluasi:",
        len(df_evaluasi)
    )

    if "MAE" in df_evaluasi.columns:

        print(
            "Rata-rata MAE:",
            df_evaluasi["MAE"].mean()
        )

    if "RMSE" in df_evaluasi.columns:

        print(
            "Rata-rata RMSE:",
            df_evaluasi["RMSE"].mean()
        )

    if "MAPE" in df_evaluasi.columns:

        print(
            "Rata-rata MAPE:",
            df_evaluasi["MAPE"].mean()
        )

    if "WAPE" in df_evaluasi.columns:

        print(
            "Rata-rata WAPE:",
            df_evaluasi["WAPE"].mean()
        )


print("\n")
print("=" * 70)
print("PROGRAM SELESAI")
print("=" * 70)
