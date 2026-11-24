"""Testes unitários para o módulo de contagem (counter.py).

Cobre:
    - mesmo ID contado apenas uma vez (sets de IDs)
    - IDs diferentes contados separadamente
    - classes contadas separadamente
    - contagem por zona com IN e OUT
    - summary serializável
"""

from src.counter import ObjectCounter
from src.detector import Detection
from src.zones import Zone, ZoneConfig


def _make_zone():
    return Zone(
        ZoneConfig(
            name="Linha Central",
            type="line",
            points=[[0, 200], [400, 200]],
            color=[0, 255, 128],
        )
    )


def _detection(track_id: int, class_name: str, y: float, class_id: int = 2):
    return Detection(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bbox=(100.0, y - 20.0, 300.0, y + 20.0),
        bottom_center=(200.0, y),
    )


def test_mesmo_id_contado_apenas_uma_vez():
    counter = ObjectCounter([_make_zone()])

    # Frame 1: ID 1 acima da linha (sem prev, não há cruzamento ainda)
    counter.update([_detection(1, "car", 100.0)], timestamp=0.0)
    # Frame 2: ID 1 cruza a linha de cima para baixo (IN)
    counter.update([_detection(1, "car", 300.0)], timestamp=1.0)
    # Frame 3: ID 1 continua abaixo (já foi contado, não duplica)
    counter.update([_detection(1, "car", 300.0)], timestamp=2.0)

    assert counter.zone_count("Linha Central") == 1
    assert counter.total_unique() == 1
    assert counter.class_count("car") == 1


def test_ids_diferentes_contados_separadamente():
    counter = ObjectCounter([_make_zone()])

    counter.update([_detection(1, "car", 100.0)], timestamp=0.0)
    counter.update([_detection(1, "car", 300.0)], timestamp=1.0)
    counter.update([_detection(2, "truck", 100.0)], timestamp=2.0)
    counter.update([_detection(2, "truck", 300.0)], timestamp=3.0)

    assert counter.zone_count("Linha Central") == 2
    assert counter.total_unique() == 2
    assert counter.class_count("car") == 1
    assert counter.class_count("truck") == 1


def test_zone_in_out_contagem():
    counter = ObjectCounter([_make_zone()])

    # ID 1: cima -> baixo = IN
    counter.update([_detection(1, "car", 100.0)], timestamp=0.0)
    counter.update([_detection(1, "car", 300.0)], timestamp=1.0)
    # ID 2: baixo -> cima = OUT
    counter.update([_detection(2, "car", 300.0)], timestamp=2.0)
    counter.update([_detection(2, "car", 100.0)], timestamp=3.0)

    ins, outs = counter.zone_in_out("Linha Central")
    assert ins == 1
    assert outs == 1
    assert counter.zone_count("Linha Central") == 2


def test_sem_zona_retorna_zero():
    counter = ObjectCounter([])
    counter.update([_detection(1, "car", 300.0)], timestamp=0.0)

    assert counter.total_unique() == 1
    assert counter.class_count("car") == 1


def test_summary_serializavel():
    counter = ObjectCounter([_make_zone()])
    counter.update([_detection(1, "car", 100.0)], timestamp=0.0)
    counter.update([_detection(1, "car", 300.0)], timestamp=1.0)

    summary = counter.summary()
    assert summary["total_unique_objects"] == 1
    assert summary["by_class"] == {"car": 1}
    assert "Linha Central" in summary["zones"]
    assert summary["zones"]["Linha Central"]["count"] == 1
    assert summary["zones"]["Linha Central"]["in"] == 1
    assert summary["zones"]["Linha Central"]["out"] == 0
    assert summary["zones"]["Linha Central"]["type"] == "line"
