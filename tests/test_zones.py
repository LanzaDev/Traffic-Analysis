"""Testes unitários para o módulo de zonas ROI (zones.py).

Cobre:
    - ponto dentro/fora/borda de polígono
    - cruzamento de linha com sentido IN e OUT
    - movimento que não cruza a linha (None)
"""

import pytest

from src.zones import Zone, ZoneConfig


def _polygon_zone():
    return Zone(
        ZoneConfig(
            name="Quadrado",
            type="polygon",
            points=[[100, 100], [300, 100], [300, 300], [100, 300]],
            color=[0, 200, 255],
        )
    )


def _line_zone():
    return Zone(
        ZoneConfig(
            name="Linha Central",
            type="line",
            points=[[0, 200], [400, 200]],
            color=[0, 255, 128],
        )
    )


class TestZonePolygonContains:
    def test_ponto_dentro_poligono(self):
        z = _polygon_zone()
        assert z.contains((200, 200)) is True

    def test_ponto_fora_poligono(self):
        z = _polygon_zone()
        assert z.contains((500, 500)) is False
        assert z.contains((0, 0)) is False

    def test_ponto_na_borda_eh_dentro(self):
        z = _polygon_zone()
        # pointPolygonTest com measureDist=False retorna >= 0 na borda.
        assert z.contains((100, 200)) is True
        assert z.contains((200, 100)) is True
        assert z.contains((300, 200)) is True
        assert z.contains((200, 300)) is True

    def test_tipo_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            ZoneConfig(name="X", type="invalido", points=[[0, 0], [1, 1]])

    def test_poligono_com_poucos_pontos_levanta_erro(self):
        with pytest.raises(ValueError):
            ZoneConfig(name="X", type="polygon", points=[[0, 0], [1, 1]])


class TestZoneLineCrossing:
    def test_cruzamento_sentido_in(self):
        z = _line_zone()
        # p1=(0,200) -> p2=(400,200). Movendo de cima (y<200) para baixo (y>200)
        # o produto vetorial indica lado negativo para positivo => "in".
        direcao = z.crossing((200, 100), (200, 300))
        assert direcao == "in"

    def test_cruzamento_sentido_out(self):
        z = _line_zone()
        direcao = z.crossing((200, 300), (200, 100))
        assert direcao == "out"

    def test_sem_cruzamento_retorna_none(self):
        z = _line_zone()
        # Ambos acima da linha
        assert z.crossing((200, 100), (300, 150)) is None
        # Ambos abaixo da linha
        assert z.crossing((200, 250), (300, 300)) is None

    def test_movimento_paralelo_nao_cruza(self):
        z = _line_zone()
        assert z.crossing((100, 200), (300, 200)) is None

    def test_zona_poligono_nao_tem_crossing(self):
        z = _polygon_zone()
        assert z.crossing((50, 50), (200, 200)) is None

    def test_zona_linha_nao_tem_contains(self):
        z = _line_zone()
        assert z.contains((200, 200)) is False
