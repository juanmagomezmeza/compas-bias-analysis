import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import Papa from 'papaparse';
import { FileText, Download, ShieldCheck, BarChart3 } from 'lucide-react';

function App() {
  const [headers, setHeaders] = useState([]);
  const [fileName, setFileName] = useState("");
  
  const { 
    register, 
    handleSubmit, 
    setValue, 
    formState: { isValid } 
  } = useForm({
    mode: "onChange", // Valida en tiempo real mientras el usuario escribe/selecciona
    defaultValues: {
      favorable_classes: [0],
      plot_mapping: { risk_threshold: 5 }
    }
  });

  // 1. Lógica para leer el CSV y extraer encabezados
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      Papa.parse(file, {
        preview: 1,
        complete: (results) => {
          const cols = results.data[0];
          setHeaders(cols);
          setValue("data_path", file.name);
        }
      });
    }
  };

  // 2. Generar y descargar el JSON
  const onSubmit = (data) => {
    const finalConfig = {
      dataset_name: data.dataset_name,
      data_path: data.data_path,
      label_name: data.label_name,
      favorable_classes: [0], 
      protected_attribute_names: [data.protected_col],
      privileged_classes: [[data.privileged_val_raw]],
      features_to_keep: headers, 
      plot_mapping: {
        target_label_col: data.label_name,
        target_label_name: data.plot_mapping.target_label_name,
        protected_col: data.protected_col,
        secondary_col: data.secondary_col,
        priv_val_name: data.privileged_val_raw,
        unpriv_val_name: data.unprivileged_val_name,
        risk_score_col: data.risk_score_col,
        risk_threshold: Number(data.plot_mapping.risk_threshold)
      }
    };
  
    const blob = new Blob([JSON.stringify(finalConfig, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `config_${data.dataset_name.replace(/\s+/g, '_')}.json`;
    link.click();
  };

  // Estilo común para inputs
  const inputBaseStyle = "w-full bg-slate-700 border-slate-600 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-emerald-500 transition-all";

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        
        <header className="flex items-center gap-4 mb-10 border-b border-slate-800 pb-6">
          <ShieldCheck className="text-emerald-400 w-10 h-10" />
          <div>
            <h1 className="text-3xl font-bold">Gobernanza Algorítmica UI</h1>
            <p className="text-slate-400 text-sm uppercase tracking-wider font-semibold">Tesis: Auditoría de Sesgos en COMPAS</p>
          </div>
        </header>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
          
          {/* SECCIÓN 1: Carga de Datos */}
          <section className="bg-slate-800 p-6 rounded-xl border border-slate-700">
            <h2 className="flex items-center gap-2 text-xl font-semibold mb-4">
              <FileText className="text-blue-400" /> 1. Carga de Dataset
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm text-slate-400 mb-2">Archivo CSV a auditar</label>
                <input 
                  type="file" 
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-2">Nombre del Proyecto</label>
                <input 
                  {...register("dataset_name", { required: true })} 
                  placeholder="Ej: Análisis COMPAS ProPublica" 
                  className={inputBaseStyle} 
                />
              </div>
            </div>
          </section>

          {/* SECCIÓN 2: Mapeo de Variables */}
          {headers.length > 0 && (
            <section className="bg-slate-800 p-6 rounded-xl border border-slate-700 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <h2 className="flex items-center gap-2 text-xl font-semibold mb-6">
                <BarChart3 className="text-emerald-400" /> 2. Mapeo de Atributos y Visualización
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 border-b border-slate-700 pb-8">
                <div>
                  <label className="block text-sm text-slate-400 mb-2 font-medium">Variable Objetivo (Label)</label>
                  <select {...register("label_name", { required: true })} className={inputBaseStyle}>
                    <option value="">Seleccione...</option>
                    {headers.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
            
                <div>
                  <label className="block text-sm text-slate-400 mb-2 font-medium">Atributo Protegido</label>
                  <select {...register("protected_col", { required: true })} className={inputBaseStyle}>
                    <option value="">Seleccione...</option>
                    {headers.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
            
                <div>
                  <label className="block text-sm text-slate-400 mb-2 font-medium">Valor Privilegiado (Manual)</label>
                  <input 
                    {...register("privileged_val_raw", { required: true })} 
                    placeholder="Ej: Caucasian" 
                    className={inputBaseStyle} 
                  />
                </div>
              </div>
            
              <div className="pt-2">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6">Configuración de Reporte y Gráficos</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                  <div>
                    <label className="block text-sm text-slate-400 mb-2 font-medium">Columna Secundaria (Filtro)</label>
                    <select {...register("secondary_col", { required: true })} className={inputBaseStyle}>
                      <option value="">Seleccione...</option>
                      {headers.map(h => <option key={h} value={h}>{h}</option>)}
                    </select>
                  </div>
            
                  <div>
                    <label className="block text-sm text-slate-400 mb-2 font-medium">Nombre Grupo No Privilegiado</label>
                    <input 
                      {...register("unprivileged_val_name", { required: true })} 
                      placeholder="Ej: African-American" 
                      className={inputBaseStyle} 
                    />
                  </div>
            
                  <div>
                    <label className="block text-sm text-slate-400 mb-2 font-medium">Columna de Riesgo (Score)</label>
                    <select {...register("risk_score_col", { required: true })} className={inputBaseStyle}>
                      <option value="">Seleccione...</option>
                      {headers.map(h => <option key={h} value={h}>{h}</option>)}
                    </select>
                  </div>
            
                  <div>
                    <label className="block text-sm text-slate-400 mb-2 font-medium">Etiqueta Eje Y (Gráfico)</label>
                    <input 
                      {...register("plot_mapping.target_label_name", { required: true })} 
                      placeholder="Ej: Reincidencia" 
                      className={inputBaseStyle} 
                    />
                  </div>
            
                  <div className="flex items-center gap-4 bg-slate-900/50 p-4 rounded-lg border border-slate-700/50 md:col-span-2">
                    <div className="flex-1">
                      <label className="block text-sm text-slate-400 font-medium">Umbral de Riesgo (Risk Threshold)</label>
                      <p className="text-xs text-slate-500">Puntaje a partir del cual se considera riesgo alto.</p>
                    </div>
                    <input 
                      type="number" 
                      {...register("plot_mapping.risk_threshold", { required: true, min: 0 })} 
                      className="bg-slate-700 p-2.5 rounded-lg w-24 text-center text-emerald-400 font-bold border-slate-600 outline-none" 
                    />
                  </div>
                </div>
              </div>
            </section>
          )}

          <button 
            type="submit" 
            disabled={!isValid || headers.length === 0}
            className="w-full flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 font-bold py-4 rounded-xl transition-all shadow-lg shadow-emerald-500/20"
          >
            <Download size={20} />
            Generar config.json
          </button>

        </form>
      </div>
    </div>
  );
}

export default App;