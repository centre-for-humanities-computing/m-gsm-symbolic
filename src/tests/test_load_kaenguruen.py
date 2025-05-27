from pydantic import TypeAdapter

from m_gsm_symbolic.kaenguruen.load_data import KaenguruenProblem, load_kaenguruen


def test_load_kaenguruen():
    # Test if the function runs without errors
    load_kaenguruen()

    ta = TypeAdapter(KaenguruenProblem)
    for sample in load_kaenguruen():
        ta.validate(sample)
