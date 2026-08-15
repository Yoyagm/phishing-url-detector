# Progreso

## Estado actual

Proyecto **terminado y publicado**. Repo público en `github.com/Yoyagm/phishing-url-detector`, demo
viva en <https://yoyagm.github.io/phishing-url-detector/>, 113 tests en verde, informe LaTeX de 9
páginas compilado con figuras y tablas generadas desde las ejecuciones reales.

Siguiente paso si se retoma: nada bloqueante. Lo que aportaría más valor es reentrenar contra una
captura nueva de los feeds para medir la degradación temporal del modelo (los números publicados
corresponden a la captura del 2026-08-14), y verificar las licencias de redistribución de las cuatro
fuentes de datos, que quedó sin comprobar.

Bloqueos: ninguno.

## Registro

### 2026-08-14

- **Hecho:** Detector de phishing por nombre de host, de punta a punta. Auditoría de fugas de
  PhiUSIIL, modelo LightGBM con split agrupado, evaluación out-of-distribution y temporal, servicio
  FastAPI dockerizado, demo estática en GitHub Pages que calcula TreeSHAP en el navegador, informe
  LaTeX y README. 113 tests.

- **Hallazgo principal:** PhiUSIIL (UCI 967) tiene **dos fugas de etiqueta independientes**.
  `URLSimilarityIndex` vale exactamente 100.000 en las 134.850 filas legítimas y en 786 de 100.945 de
  phishing — una regla de una línea da 0,9967 de accuracy. La segunda es peor: la clase legítima es
  un artefacto de recolección (100 % `https://`, 100 % `www.`, **0,00 %** con path, query o `@`),
  así que "tiene path → phishing" acierta 0,6935 sin fallar nunca sobre una legítima. Esa segunda
  fuga **sobrevive a tirar las 54 columnas precalculadas y entrenar desde la URL cruda**, que era el
  rescate obvio: features léxicas a mano siguen dando 0,9951 sin medir nada.

- **Decisión:** Entrenar **solo sobre el nombre de host canonicalizado**. Se descartó entrenar sobre
  la URL completa (fuga 2) y usar las features de contenido HTML (exigen descargar la página, que es
  imposible antes de que el usuario haga clic). Hay un test guardián,
  `tests/test_no_host_leakage.py`, que muta esquema, path, query, puerto y `www.` y exige que el
  vector de features no cambie ni un bit, con control negativo que planta una fuga para comprobar
  que el guardián sabe fallar.

- **Decisión:** Agrupar el split por dominio registrable usando **solo la sección ICANN** de la
  Public Suffix List. Con la sección privada, `evil.vercel.app` y `nice.vercel.app` serían grupos
  distintos y el modelo cobraría por memorizar un proveedor de hosting. La sección privada pasó a ser
  la feature `is_platform_hosted`.

- **Decisión:** Demo en **GitHub Pages estático** con scoring y TreeSHAP reimplementados en
  JavaScript. Se descartó Gradio/HF Spaces (de pago desde 2026) y Streamlit Community Cloud (duerme,
  y un reclutador se encontraría una pantalla de "despertar la app"). El riesgo del puerto se cubre
  con un test de paridad sobre 567 URLs: features y raw scores **idénticos bit a bit**, atribuciones
  SHAP con desviación máxima 8,9·10⁻¹⁶.

- **Resultado honesto:** del **1.0000 de accuracy** publicado a **70,5 % de recall con 1 % de falsos
  positivos** fuera de distribución (ROC-AUC 0,9372), y 58,5 % de detección sobre un feed de
  OpenPhish capturado el mismo día. Elegir el umbral por ROC en vez del 0,5 por defecto baja los
  falsos positivos de 13.453 a 999 (−92,6 %) a cambio de 18 puntos de recall.

- **Corrección:** Mi conteo inicial de 815 filas de phishing con índice ≥ 99,9 estaba mal; venía de
  un `awk` que partía mal las filas con comas entrecomilladas. El número correcto, con pandas, es
  **786**.

- **Bloqueo (no resuelto, mitigado):** No se llegó a verificar los términos de redistribución de
  PhiUSIIL, Tranco, OpenPhish y PhishTank. Se optó por lo conservador igualmente: **ningún dato crudo
  está commiteado**; `scripts/fetch_data.py` los descarga.

- **Limitación del entorno:** La delegación en subagentes no llegó a ejecutarse en ninguno de los
  cinco intentos, incluida una sonda trivial. Las fases de diseño, construcción, prueba, revisión y
  documentación se hicieron en el hilo principal.
