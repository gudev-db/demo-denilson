import streamlit as st
import cv2
import math
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Constantes
DENSIDADE_MADEIRA = 600  # kg/m³ (ajuste para o tipo de madeira)
COMPRIMENTO_TORA = 2.0    # metros (valor assumido)
FATOR_CONVERSAO = 0.002  # Fator de calibração (500px = 1m)

# Carrega o modelo (com cache para evitar recarregar)
@st.cache_resource
def carregar_modelo():
    return YOLO('yolov8n.pt')  # Substitua por seu modelo customizado

def estimar_massa(imagem):
    """Processa a imagem e retorna a imagem anotada com as estimativas"""
    modelo = carregar_modelo()
    resultados = modelo(imagem)
    
    imagem_anotada = imagem.copy()
    previsoes = []
    
    for det in resultados[0].boxes:
        # Obtém as coordenadas da caixa delimitadora
        x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
        
        # Calcula diâmetro e massa
        largura_px = x2 - x1
        diametro_m = largura_px * FATOR_CONVERSAO
        raio = diametro_m / 2
        volume = math.pi * (raio ** 2) * COMPRIMENTO_TORA
        massa_kg = DENSIDADE_MADEIRA * volume
        
        # Armazena a previsão
        previsoes.append({
            'bbox': [x1, y1, x2, y2],
            'diametro_m': diametro_m,
            'massa_kg': massa_kg,
            'confianca': det.conf.item()
        })
        
        # Desenha a caixa delimitadora
        cv2.rectangle(imagem_anotada, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Cria o texto da etiqueta
        texto = f"{massa_kg:.1f}kg | {diametro_m:.2f}m | {det.conf.item():.2f}"
        
        # Calcula o tamanho do texto
        (largura_texto, altura_texto), _ = cv2.getTextSize(
            texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        # Desenha o fundo da etiqueta
        cv2.rectangle(
            imagem_anotada,
            (x1, y1 - altura_texto - 10),
            (x1 + largura_texto, y1),
            (0, 255, 0),
            -1
        )
        
        # Adiciona o texto
        cv2.putText(
            imagem_anotada,
            texto,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    return imagem_anotada, previsoes

# Interface do Streamlit
st.title("📊 Estimador de Massa de Toras de Madeira")
st.markdown("Faça upload de uma imagem de toras para obter estimativas de massa")

arquivo = st.file_uploader(
    "Selecione uma imagem...",
    type=["jpg", "jpeg", "png"]
)

if arquivo is not None:
    # Lê a imagem
    imagem = np.array(Image.open(arquivo))
    
    # Mostra a original
    st.subheader("Imagem Original")
    st.image(imagem, use_column_width=True)
    
    # Processa a imagem
    with st.spinner("Detectando toras e calculando massas..."):
        imagem_anotada, previsoes = estimar_massa(imagem)
        
        # Mostra resultados
        st.subheader("Resultados da Análise")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(imagem_anotada, use_column_width=True, caption="Imagem Anotada")
        
        with col2:
            st.subheader(f"Toras Detectadas: {len(previsoes)}")
            for i, prev in enumerate(previsoes, 1):
                st.markdown(
                    f"""
                    **Tora {i}**
                    - Massa: {prev['massa_kg']:.1f} kg
                    - Diâmetro: {prev['diametro_m']:.2f} m
                    - Confiança: {prev['confianca']:.2f}
                    """
                )
            
            # Estatísticas resumidas
            if previsoes:
                massa_total = sum(p['massa_kg'] for p in previsoes)
                diametro_medio = sum(p['diametro_m'] for p in previsoes) / len(previsoes)
                
                st.success(
                    f"**Massa Total:** {massa_total:.1f} kg\n\n"
                    f"**Diâmetro Médio:** {diametro_medio:.2f} m"
                )

# Barra lateral com instruções
with st.sidebar:
    st.header("Instruções")
    st.markdown("""
    1. Faça upload de uma imagem contendo toras
    2. Aguarde o processamento
    3. Visualize as estimativas
    
    **Dicas:**
    - Use imagens bem iluminadas
    - Posicione a câmera perpendicular às toras
    - Para maior precisão, calibre:
        - `FATOR_CONVERSAO`
        - `DENSIDADE_MADEIRA`
    """)
    
    st.markdown("---")
    st.markdown("**Configurações**")
    tipo_madeira = st.selectbox(
        "Densidade da Madeira",
        ("Pinho (500 kg/m³)", "Carvalho (700 kg/m³)", "Personalizado")
    )
    
    if tipo_madeira == "Personalizado":
        densidade_personalizada = st.number_input(
            "Informe a densidade (kg/m³)",
            value=600,
            min_value=100,
            max_value=1000
        )
        DENSIDADE_MADEIRA = densidade_personalizada
    else:
        DENSIDADE_MADEIRA = 500 if "Pinho" in tipo_madeira else 700
