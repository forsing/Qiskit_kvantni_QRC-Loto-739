#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Qiskit Quantum Regresija

Kvantna obrada istorije i kvantni regresioni izlaz.
Dozvoljeni klasični optimizator: L-BFGS-B.

ISPRAVLJENA RASPODELA:

    P(b1,...,b7 | H)
        = proizvod_r w_r(br | H) / Z(H),

za sve:
    1 <= b1 < b2 < ... < b7 <= 39.

Z(H) sabira težine SVIH legalnih kombinacija.
Nema uzastopne normalizacije na sve užem skupu brojeva.

Sedam izlaznih kvantnih kola imaju zasebne parametre.
Ukupno: 7 * 3 * 6 = 126 učenih kvantnih uglova.

Učenje minimizuje kvadratnu grešku kumulativnih raspodela
pozicija, dobijenih iz ZAJEDNIČKE normalizovane raspodele.

Qiskit obrađuje istoriju tačnim vektorima stanja.
PyTorch diferencira tačnu simulaciju izlaznih kvantnih kola.
Ne koristi se klasična neuronska mreža niti klasični regresor.

Dinamičko programiranje računa:
- normalizaciju cele kombinacije;
- marginalne raspodele svih sedam pozicija;
- tačan maksimum definisanog modela.

Nema:
- slučajne inicijalizacije ili uzorkovanja;
- klasifikatora;
- nametnute normalne ili uniformne raspodele;
- filtera parnosti, zona, zbira, razmaka ili uzastopnih brojeva;
- zaokruživanja regresiranih brojeva;
- prisilnog razlikovanja rezultata različitih CSV-ova. 

CSV: bez zaglavlja; sedam različitih celih brojeva 1–39.
Redovi ostaju u izvornom, hronološkom redosledu.
"""


import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import itertools
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import scipy
    from scipy.optimize import minimize
    import torch
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector
except (ImportError, OSError) as exc:
    raise SystemExit(
        "Zavisnosti se ne mogu učitati.\n"
        "Instalacija:\n"
        "python3 -m pip install numpy scipy torch qiskit\n"
        f"Detalj: {exc}"
    )

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

DEFAULT_FILES = (
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_2974.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_plus_1730.csv",
)

BALLS = 39
PICKS = 7
QUBITS = 6
DIM = 2 ** QUBITS

RESERVOIR_LAYERS = 3
READOUT_LAYERS = 3
PARAMETER_COUNT = PICKS * READOUT_LAYERS * QUBITS

BATCH_DRAWS = 96
MAX_ITERATIONS = 60
PATIENCE = 10

REAL = torch.float64
COMPLEX = torch.complex128

RING = tuple((q, (q + 1) % QUBITS) for q in range(QUBITS))

basis = np.arange(DIM)
cz_sign = np.ones(DIM, dtype=np.float64)

for first, second in RING:
    both = ((basis >> first) & 1) & ((basis >> second) & 1)
    cz_sign *= 1.0 - 2.0 * both

CZ_SIGN = torch.tensor(cz_sign, dtype=REAL).reshape(1, 1, DIM)

# Samo ograničenja koja neposredno slede iz pravila 7/39.
support = np.zeros((PICKS, BALLS), dtype=np.float64)

for rank in range(PICKS):
    support[rank, rank:BALLS - PICKS + rank + 1] = 1.0

SUPPORT = torch.tensor(support, dtype=REAL)[None, :, :]


def read_csv(path):
    path = Path(path).expanduser()

    if not path.is_file():
        raise ValueError(f"CSV ne postoji: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            fields = re.split(r"[,;\s]+", line)

            if len(fields) != PICKS:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 polja, pronađeno {len(fields)}."
                )

            numbers = []

            for field in fields:
                token = field.strip().strip('"').strip("'")

                if not re.fullmatch(r"[+-]?\d+", token):
                    raise ValueError(
                        f"{path.name}, red {line_number}: "
                        f"{field!r} nije ceo broj."
                    )

                numbers.append(int(token))

            if any(n < 1 or n > BALLS for n in numbers):
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "broj izvan raspona 1–39."
                )

            if len(set(numbers)) != PICKS:
                raise ValueError(
                    f"{path.name}, red {line_number}: ponovljen broj."
                )

            rows.append(sorted(numbers))

    if not rows:
        raise ValueError(f"Prazan CSV: {path}")

    return np.asarray(rows, dtype=np.int64)


def format_numbers(numbers):
    return ", ".join(str(int(n)) for n in numbers)


def longest_run(numbers):
    best = current = 1

    for previous, number in zip(numbers[:-1], numbers[1:]):
        current = current + 1 if number == previous + 1 else 1
        best = max(best, current)

    return best


def initial_parameters():
    index = np.arange(1, PARAMETER_COUNT + 1, dtype=np.float64)

    return (
        0.85 * np.sin(np.sqrt(2.0) * index)
        + 0.31 * np.cos(np.sqrt(3.0) * index)
    )


def rank_encoding_circuit(rank):
    circuit = QuantumCircuit(QUBITS)

    for qubit in range(QUBITS):
        circuit.ry(
            np.pi * (rank + 1) * (qubit + 1) / (PICKS * QUBITS),
            qubit,
        )
        circuit.rz(
            np.pi * (rank + 1) / (PICKS * (qubit + 1)),
            qubit,
        )

    return circuit


def readout_circuit(angles):
    circuit = QuantumCircuit(QUBITS)

    for layer in range(READOUT_LAYERS):
        for qubit in range(QUBITS):
            circuit.ry(float(angles[layer, qubit]), qubit)

        if layer + 1 < READOUT_LAYERS:
            for first, second in RING:
                circuit.cz(first, second)

    return circuit


class QuantumHistory:
    def __init__(self):
        reservoir = QuantumCircuit(QUBITS)

        for _ in range(RESERVOIR_LAYERS):
            for qubit in range(QUBITS):
                reservoir.rx(np.pi / 4.0, qubit)

            for first, second in RING:
                reservoir.cz(first, second)

            for qubit in range(QUBITS):
                reservoir.ry(np.pi / 6.0, qubit)

        self.reservoir = Operator(reservoir)

        self.rank_operators = [
            np.asarray(
                Operator(rank_encoding_circuit(rank)).data,
                dtype=np.complex128,
            )
            for rank in range(PICKS)
        ]

    @staticmethod
    def encode_draw(draw):
        circuit = QuantumCircuit(QUBITS)

        for rank, number in enumerate(draw):
            value = (float(number) - 1.0) / (BALLS - 1.0)

            circuit.ry(np.pi * value, rank % QUBITS)
            circuit.rz(
                np.pi * value / 2.0,
                (rank + 1) % QUBITS,
            )

        return circuit

    def transform(self, draws):
        states = np.empty(
            (len(draws), DIM), dtype=np.complex128
        )
        state = Statevector.from_label("0" * QUBITS)

        for index, draw in enumerate(draws):
            state = state.evolve(self.encode_draw(draw))
            state = state.evolve(self.reservoir)

            norm = float(np.vdot(state.data, state.data).real)

            if not np.isfinite(norm) or abs(norm - 1.0) > 1e-8:
                raise ArithmeticError(
                    f"Neispravno kvantno stanje u kolu {index + 1}."
                )

            states[index] = state.data

            if (index + 1) % 1000 == 0:
                print(
                    f"  Qiskit istorija: {index + 1}/{len(draws)}",
                    flush=True,
                )

        return states

    def encode_ranks(self, states):
        states = np.asarray(states, dtype=np.complex128)

        if states.ndim == 1:
            states = states[None, :]

        # Qiskit matrice deluju na vektore stanja.
        encoded = np.stack(
            [states @ operator.T for operator in self.rank_operators],
            axis=1,
        )

        return torch.from_numpy(np.ascontiguousarray(encoded))


def quantum_readout(states, parameters):
    """
    Tačna diferencijabilna simulacija sedam kvantnih kola.

    states:     [batch, 7, 64], kompleksno
    parameters: [126], realno
    """
    angles = parameters.reshape(
        PICKS, READOUT_LAYERS, QUBITS
    )

    output = states
    batch_size = states.shape[0]

    for layer in range(READOUT_LAYERS):
        for qubit in range(QUBITS):
            shaped = output.reshape(
                batch_size,
                PICKS,
                2 ** (QUBITS - qubit - 1),
                2,
                2 ** qubit,
            )

            zero = shaped[:, :, :, 0, :]
            one = shaped[:, :, :, 1, :]

            angle = angles[:, layer, qubit]
            cosine = torch.cos(angle / 2.0)[None, :, None, None]
            sine = torch.sin(angle / 2.0)[None, :, None, None]

            output = torch.stack(
                (
                    cosine * zero - sine * one,
                    sine * zero + cosine * one,
                ),
                dim=3,
            ).reshape(batch_size, PICKS, DIM)

        if layer + 1 < READOUT_LAYERS:
            output = output * CZ_SIGN

    return output


def quantum_weights(states, parameters):
    output = quantum_readout(states, parameters)

    probabilities = output.real.square() + output.imag.square()

    # Težine predstavljaju Bornove verovatnoće kvantnih izlaza.
    # Ovde se NE normalizuju po prethodno izabranom broju.
    return probabilities[:, :, :BALLS] * SUPPORT


def normalize_vector(values):
    normalizer = values.sum(dim=-1, keepdim=True)

    if (
        not bool(torch.isfinite(normalizer).all())
        or bool((normalizer <= 0.0).any())
    ):
        raise ArithmeticError(
            "Zajednička raspodela nema konačnu pozitivnu masu."
        )

    return values / normalizer, normalizer


def exclusive_prefix_sum(values):
    summed = torch.cumsum(values, dim=-1)

    return torch.cat(
        (torch.zeros_like(summed[:, :1]), summed[:, :-1]),
        dim=-1,
    )


def exclusive_suffix_sum(values):
    summed = torch.flip(
        torch.cumsum(torch.flip(values, dims=(-1,)), dim=-1),
        dims=(-1,),
    )

    return torch.cat(
        (summed[:, 1:], torch.zeros_like(summed[:, :1])),
        dim=-1,
    )


def joint_marginals(weights):
    """
    Tačna zajednička normalizacija svih strogo rastućih putanja.

    Vraća:
        marginalne raspodele [batch, pozicija, broj],
        logaritam zajedničke normalizacione konstante.

    Skaliranje međurezultata služi numeričkoj stabilnosti.
    Ne zavisi od pojedinačne izabrane putanje.
    """
    positions = weights.shape[1]

    forward = []

    current, scale = normalize_vector(weights[:, 0, :])
    forward.append(current)
    log_partition = torch.log(scale[:, 0])

    for rank in range(1, positions):
        current = (
            weights[:, rank, :] * exclusive_prefix_sum(current)
        )
        current, scale = normalize_vector(current)

        forward.append(current)
        log_partition = log_partition + torch.log(scale[:, 0])

    backward = [None] * positions
    backward[-1] = torch.ones_like(weights[:, -1, :])

    for rank in range(positions - 2, -1, -1):
        current = exclusive_suffix_sum(
            weights[:, rank + 1, :] * backward[rank + 1]
        )
        current, _ = normalize_vector(current)
        backward[rank] = current

    marginals = []

    for rank in range(positions):
        marginal, _ = normalize_vector(
            forward[rank] * backward[rank]
        )
        marginals.append(marginal)

    return torch.stack(marginals, dim=1), log_partition


class Dataset:
    def __init__(self, states, outcomes, history_engine):
        # Nijedan broj ciljne kombinacije ne ulazi u kvantni ulaz.
        self.states = history_engine.encode_ranks(states)
        self.outcomes = np.asarray(outcomes, dtype=np.int64)

        targets = (
            outcomes[:, :, None]
            <= np.arange(1, BALLS)[None, None, :]
        ).astype(np.float64)

        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.outcomes)


def dataset_loss(data, start, stop, flat_parameters, gradient=False):
    count = stop - start

    if count <= 0:
        raise ValueError("Prazan skup za učenje ili proveru.")

    parameters = torch.tensor(
        np.asarray(flat_parameters),
        dtype=REAL,
        requires_grad=gradient,
    )

    total = 0.0

    with torch.set_grad_enabled(gradient):
        for first in range(start, stop, BATCH_DRAWS):
            last = min(first + BATCH_DRAWS, stop)
            fraction = (last - first) / count

            weights = quantum_weights(
                data.states[first:last], parameters
            )
            marginals, _ = joint_marginals(weights)

            cumulative = torch.cumsum(
                marginals, dim=-1
            )[:, :, :BALLS - 1]

            loss = (
                (cumulative - data.targets[first:last]).square().mean()
                * fraction
            )

            if not bool(torch.isfinite(loss)):
                raise ArithmeticError("Nekonačna regresiona greška.")

            total += float(loss.detach())

            if gradient:
                loss.backward()

    if gradient:
        if parameters.grad is None:
            raise ArithmeticError("Nedostaje gradijent kvantnih uglova.")

        derivative = parameters.grad.detach().numpy().copy()

        if not np.all(np.isfinite(derivative)):
            raise ArithmeticError("Nekonačan gradijent.")

        return total, derivative

    return total


def maximum_path(weights):
    """Tačan maksimum proizvoda težina legalne kombinacije."""
    weights = np.asarray(weights, dtype=np.float64)
    positions, numbers = weights.shape

    logs = np.full_like(weights, -np.inf)
    np.log(weights, out=logs, where=weights > 0.0)

    scores = logs[0].copy()
    back = np.full(
        (positions, numbers), -1, dtype=np.int64
    )

    for rank in range(1, positions):
        updated = np.full(numbers, -np.inf)

        best_score = -np.inf
        best_previous = -1

        for number in range(numbers):
            if number > 0 and scores[number - 1] > best_score:
                best_score = scores[number - 1]
                best_previous = number - 1

            if np.isfinite(best_score) and np.isfinite(logs[rank, number]):
                updated[number] = best_score + logs[rank, number]
                back[rank, number] = best_previous

        scores = updated

    last = int(np.argmax(scores))

    if not np.isfinite(scores[last]):
        raise ArithmeticError("Nema legalne putanje modela.")

    path = [last]

    for rank in range(positions - 1, 0, -1):
        last = int(back[rank, last])

        if last < 0:
            raise ArithmeticError("Neispravna rekonstrukcija putanje.")

        path.append(last)

    return tuple(reversed(path))


def predict(history_state, parameters, history_engine):
    encoded = history_engine.encode_ranks(history_state)
    parameter_tensor = torch.tensor(parameters, dtype=REAL)

    with torch.no_grad():
        weights = quantum_weights(encoded, parameter_tensor)
        marginals, log_partition = joint_marginals(weights)

    weight_array = weights[0].numpy()
    indices = maximum_path(weight_array)
    combination = tuple(index + 1 for index in indices)

    if (
        len(combination) != PICKS
        or len(set(combination)) != PICKS
        or tuple(sorted(combination)) != combination
        or combination[0] < 1
        or combination[-1] > BALLS
    ):
        raise ArithmeticError("Nelegalan rezultat za 7/39.")

    # Normalizovana log-masa kombinacije unutar modela.
    log_mass = float(
        sum(np.log(weight_array[r, n]) for r, n in enumerate(indices))
        - log_partition[0].item()
    )

    return combination, marginals[0].numpy(), log_mass


def verify_joint_distribution():
    """
    Provera DP-a potpunim nabrajanjem malog prostora.
    Koriste se analitički određene nejednake težine.
    """
    positions, numbers = 3, 8

    index = np.arange(1, positions * numbers + 1).reshape(
        positions, numbers
    )
    weights = 0.2 + np.cos(index * np.sqrt(2.0)) ** 2

    reference = np.zeros_like(weights)
    partition = 0.0
    best_path = None
    best_weight = -1.0

    for combination in itertools.combinations(range(numbers), positions):
        weight = float(
            np.prod([weights[r, n] for r, n in enumerate(combination)])
        )
        partition += weight

        for rank, number in enumerate(combination):
            reference[rank, number] += weight

        if weight > best_weight:
            best_weight = weight
            best_path = combination

    reference /= partition

    tensor = torch.tensor(weights[None, :, :], dtype=REAL)

    with torch.no_grad():
        marginals, log_partition = joint_marginals(tensor)

    if not np.allclose(
        marginals[0].numpy(), reference, atol=1e-12, rtol=1e-11
    ):
        raise ArithmeticError("Provera zajedničkih marginala nije prošla.")

    if not np.isclose(
        log_partition[0].item(), np.log(partition),
        atol=1e-12, rtol=1e-11,
    ):
        raise ArithmeticError("Provera normalizacije nije prošla.")

    if maximum_path(weights) != best_path:
        raise ArithmeticError("Provera maksimuma nije prošla.")


def verify_quantum_circuit(history_state, encoded_states, flat_parameters):
    angles = np.asarray(flat_parameters).reshape(
        PICKS, READOUT_LAYERS, QUBITS
    )

    with torch.no_grad():
        output = quantum_readout(
            encoded_states[:1],
            torch.tensor(flat_parameters, dtype=REAL),
        ).numpy()[0]

    for rank in range(PICKS):
        native = Statevector(history_state)
        native = native.evolve(rank_encoding_circuit(rank))
        native = native.evolve(readout_circuit(angles[rank]))

        if not np.allclose(
            native.data, output[rank], atol=1e-11, rtol=1e-10
        ):
            raise ArithmeticError(
                f"Qiskit provera nije prošla za poziciju {rank + 1}."
            )


def verify_gradient(data):
    parameters = initial_parameters()
    stop = min(3, len(data))

    _, analytic = dataset_loss(
        data, 0, stop, parameters, gradient=True
    )

    # Obuhvaćeni su različiti izlazi, slojevi i kubiti.
    indices = (0, 5, 12, 23, 48, 71, 90, 125)
    epsilon = 1e-6

    for index in indices:
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon

        numerical = (
            dataset_loss(data, 0, stop, plus)
            - dataset_loss(data, 0, stop, minus)
        ) / (2.0 * epsilon)

        if not np.isclose(
            analytic[index], numerical, atol=2e-7, rtol=3e-5
        ):
            raise ArithmeticError(
                f"Provera gradijenta nije prošla, parametar {index}."
            )


class EarlyStop(Exception):
    pass


def fit(data, train_stop, iterations, validation=None):
    initial = initial_parameters()

    state = {
        "current": 0,
        "best_loss": np.inf,
        "best_parameters": None,
        "best_iteration": 0,
        "stale": 0,
    }

    def objective(parameters):
        return dataset_loss(
            data, 0, train_stop, parameters, gradient=True
        )

    def callback(parameters):
        state["current"] += 1
        iteration = state["current"]

        if validation is not None:
            start, stop = validation
            error = dataset_loss(data, start, stop, parameters)

            if error < state["best_loss"] - 1e-10:
                state["best_loss"] = error
                state["best_parameters"] = parameters.copy()
                state["best_iteration"] = iteration
                state["stale"] = 0
            else:
                state["stale"] += 1

            if iteration == 1 or iteration % 5 == 0:
                print(
                    f"    Iteracija {iteration}/{iterations}: "
                    f"validaciona CDF-MSE={error:.8f}",
                    flush=True,
                )

            if state["stale"] >= PATIENCE:
                raise EarlyStop()

        elif iteration == 1 or iteration % 5 == 0:
            print(
                f"    Iteracija {iteration}/{iterations}",
                flush=True,
            )

    try:
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-np.pi, np.pi)] * PARAMETER_COUNT,
            callback=callback,
            options={
                "maxiter": int(iterations),
                "maxls": 30,
                "maxcor": 10,
                "ftol": 1e-12,
                "gtol": 1e-9,
            },
        )
    except EarlyStop:
        if validation is None:
            raise

        return (
            state["best_parameters"],
            state["best_iteration"],
            state["best_loss"],
        )

    if result.status not in (0, 1):
        raise ArithmeticError(
            f"Optimizacija nije završena uredno: {result.message}"
        )

    if not np.all(np.isfinite(result.x)):
        raise ArithmeticError("Neispravni naučeni kvantni uglovi.")

    if validation is not None:
        if state["best_parameters"] is None:
            raise ArithmeticError(
                "Optimizator nije napravio korak učenja."
            )

        return (
            state["best_parameters"],
            state["best_iteration"],
            state["best_loss"],
        )

    print(
        f"    Prihvaćenih koraka: {int(result.nit)}",
        flush=True,
    )

    return result.x.copy(), int(result.nit), float(result.fun)


def process_file(path, history_engine, validation_size, test_size):
    draws = read_csv(path)
    transitions = len(draws) - 1
    train_stop = transitions - validation_size - test_size
    test_start = transitions - test_size

    if train_stop < 128:
        raise ValueError(
            f"{Path(path).name}: potrebno najmanje "
            f"{129 + validation_size + test_size} kola."
        )

    fingerprint = hashlib.sha256(
        np.asarray(draws, dtype="<i8").tobytes()
    ).hexdigest()

    print("\n" + "=" * 70)
    print(Path(path).name)
    print(f"Kola: {len(draws)}")
    print(f"SHA256: {fingerprint}")
    print(
        "Kontrola: prethodno izvučeno kolo: "
        + format_numbers(draws[-1])
    )
    print(
        f"Prelazi: trening {train_stop}, "
        f"validacija {validation_size}, test {test_size}",
        flush=True,
    )

    # Novo početno kvantno stanje za svaki CSV.
    history = history_engine.transform(draws)

    # history[t] sadrži samo kola zaključno sa t.
    # Cilj tog ulaza jeste draws[t+1].
    data = Dataset(history[:-1], draws[1:], history_engine)

    verify_quantum_circuit(
        history[0], data.states[:1], initial_parameters()
    )
    verify_gradient(data)

    print(
        "  Provere prošle: Qiskit izlaz i gradijent.",
        flush=True,
    )

    print("  Izbor trajanja učenja:", flush=True)

    _, chosen_iterations, validation_error = fit(
        data,
        train_stop,
        MAX_ITERATIONS,
        validation=(train_stop, test_start),
    )

    print(
        f"  Izabrano: {chosen_iterations} iteracija; "
        f"validaciona CDF-MSE={validation_error:.8f}",
        flush=True,
    )

    print("  Učenje za nezavisnu test proveru:", flush=True)

    test_parameters, _, _ = fit(
        data, test_start, chosen_iterations
    )

    test_error = dataset_loss(
        data, test_start, transitions, test_parameters
    )

    hit_counts = np.zeros(PICKS + 1, dtype=np.int64)
    previous_hits = []

    for index in range(test_start, transitions):
        combination, _, _ = predict(
            history[index], test_parameters, history_engine
        )
        actual = set(int(n) for n in draws[index + 1])

        hits = len(set(combination) & actual)
        hit_counts[hits] += 1

        previous_hits.append(
            len(set(int(n) for n in draws[index]) & actual)
        )

    average_hits = float(
        np.dot(np.arange(PICKS + 1), hit_counts) / test_size
    )

    # Dijagnostičko poređenje sa prenošenjem prethodnog kola.
    previous_cdf = (
        draws[test_start:transitions, :, None]
        <= np.arange(1, BALLS)[None, None, :]
    )
    actual_cdf = (
        draws[test_start + 1:, :, None]
        <= np.arange(1, BALLS)[None, None, :]
    )
    previous_cdf_error = float(
        np.mean(
            (
                previous_cdf.astype(np.float64)
                - actual_cdf.astype(np.float64)
            ) ** 2
        )
    )

    print(f"Test: zajednička marginalna CDF-MSE={test_error:.8f}")
    print(f"Test: prosečan broj pogodaka={average_hits:.4f}")
    print(
        "Test: raspodela pogodaka: "
        + ", ".join(
            f"{hits} → {int(value)}"
            for hits, value in enumerate(hit_counts)
        )
    )
    print(
        "Dijagnostika prethodnog kola: "
        f"CDF-MSE={previous_cdf_error:.8f}; "
        f"prosečni pogoci={np.mean(previous_hits):.4f}"
    )

    print(
        f"  Završno učenje: svih {transitions} istorijskih prelaza",
        flush=True,
    )

    final_parameters, _, _ = fit(
        data, transitions, chosen_iterations
    )

    final_encoded = history_engine.encode_ranks(history[-1])

    verify_quantum_circuit(
        history[-1], final_encoded, final_parameters
    )

    combination, marginals, log_mass = predict(
        history[-1], final_parameters, history_engine
    )

    print("NEXT: " + format_numbers(combination))
    print(
        "Log-masa NEXT unutar naučene zajedničke raspodele: "
        f"{log_mass:.8f}"
    )
    print(
        f"Dijagnostika: najduži uzastopni niz={longest_run(combination)}; "
        f"brojeva 1–19={sum(n <= 19 for n in combination)}; "
        f"brojeva 20–39={sum(n >= 20 for n in combination)}"
    )

    print("Marginalni modovi pozicija, samo dijagnostika:")
    print(
        "  "
        + format_numbers(np.argmax(marginals, axis=1) + 1)
    )

    occurrences = int(
        np.sum(np.all(draws == np.asarray(combination), axis=1))
    )
    print(f"Ista kombinacija u istoriji: {occurrences}")

    return str(Path(path).expanduser()), combination, fingerprint


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Kvantna regresija zajedničke raspodele za Loto 7/39."
        )
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help="Tačno tri CSV putanje; inače DEFAULT_FILES.",
    )
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--test", type=int, default=64)

    args = parser.parse_args()

    if args.csv and len(args.csv) != 3:
        parser.error("Navedi tačno tri CSV putanje.")

    if args.validation < 16 or args.test < 16:
        parser.error("Validacija i test moraju imati najmanje 16 kola.")

    return args


def main():
    args = parse_args()
    files = tuple(args.csv) if args.csv else DEFAULT_FILES

    print("QISKIT — KVANTNA REGRESIJA ZAJEDNIČKE RASPODELE")
    print(
        f"Qiskit {qiskit.__version__}; "
        f"NumPy {np.__version__}; "
        f"SciPy {scipy.__version__}; "
        f"PyTorch {torch.__version__}"
    )
    print(
        f"Sedam kvantnih izlaza; "
        f"{PARAMETER_COUNT} učenih uglova; "
        "bez uzorkovanja.",
        flush=True,
    )

    verify_joint_distribution()
    print(
        "Provera zajedničke raspodele i tačnog maksimuma prošla.",
        flush=True,
    )

    history_engine = QuantumHistory()
    results = []
    failed = False

    for path in files:
        try:
            results.append(
                process_file(
                    path,
                    history_engine,
                    args.validation,
                    args.test,
                )
            )
        except Exception as exc:
            failed = True
            print(
                f"\nGREŠKA — {Path(path).name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if results:
        print("\n" + "=" * 70)
        print("ZAVRŠNE NEXT KOMBINACIJE")

        for path, combination, _ in results:
            print(f"{Path(path).name}: {format_numbers(combination)}")

        for first in range(len(results)):
            for second in range(first + 1, len(results)):
                path_a, next_a, hash_a = results[first]
                path_b, next_b, hash_b = results[second]

                if hash_a == hash_b:
                    print(
                        "Dijagnostika: isti numerički sadržaj CSV-a: "
                        f"{Path(path_a).name} / {Path(path_b).name}"
                    )
                elif next_a == next_b:
                    print(
                        "Dijagnostika: različiti CSV sadržaji imaju "
                        "isti maksimum naučenog modela: "
                        f"{Path(path_a).name} / {Path(path_b).name}"
                    )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())



"""
QISKIT — KVANTNA REGRESIJA ZAJEDNIČKE RASPODELE
Qiskit 1.4.4; NumPy 1.26.4; SciPy 1.17.1; PyTorch 2.12.0
Sedam kvantnih izlaza; 126 učenih uglova; bez uzorkovanja.
Provera zajedničke raspodele i tačnog maksimuma prošla.


======================================================================
loto7_4704_k83.csv
Kola: 4704
SHA256: 1badcf2632a9f9669e5a46a612df4e4513ab4bb0c3d488b7d746e25d458ab22c
Kontrola: prethodno izvučeno kolo: 7, 10, 13, 27, 28, 31, 38
Prelazi: trening 4575, validacija 64, test 64
  Qiskit istorija: 1000/4704
  Qiskit istorija: 2000/4704
  Qiskit istorija: 3000/4704
  Qiskit istorija: 4000/4704
  Provere prošle: Qiskit izlaz i gradijent.
  Izbor trajanja učenja:
    Iteracija 1/60: validaciona CDF-MSE=0.07983718
    Iteracija 5/60: validaciona CDF-MSE=0.07575284
    Iteracija 10/60: validaciona CDF-MSE=0.07803299
  Izabrano: 3 iteracija; validaciona CDF-MSE=0.07418958
  Učenje za nezavisnu test proveru:
    Iteracija 1/3
    Prihvaćenih koraka: 3
Test: zajednička marginalna CDF-MSE=0.08152451
Test: prosečan broj pogodaka=1.2344
Test: raspodela pogodaka: 0 → 16, 1 → 23, 2 → 20, 3 → 4, 4 → 1, 5 → 0, 6 → 0, 7 → 0
Dijagnostika prethodnog kola: CDF-MSE=0.14461936; prosečni pogoci=1.1406
  Završno učenje: svih 4703 istorijskih prelaza
    Iteracija 1/3
    Prihvaćenih koraka: 3
NEXT: 2, 13, 21, 24, 27, 29, 31
Log-masa NEXT unutar naučene zajedničke raspodele: -7.99275707
Dijagnostika: najduži uzastopni niz=1; brojeva 1–19=2; brojeva 20–39=5
Marginalni modovi pozicija, samo dijagnostika:
  2, 13, 21, 24, 27, 29, 39
Ista kombinacija u istoriji: 0



======================================================================
loto7_4704_k83_loto_2974.csv
Kola: 2974
SHA256: 992bc400ed1af4f4d5ce2395fcf52f92b086402a37ae6aff96c153c96e4a33e5
Kontrola: prethodno izvučeno kolo: 9, 14, 17, 19, 24, 28, 35
Prelazi: trening 2845, validacija 64, test 64
  Qiskit istorija: 1000/2974
  Qiskit istorija: 2000/2974
  Provere prošle: Qiskit izlaz i gradijent.
  Izbor trajanja učenja:
    Iteracija 1/60: validaciona CDF-MSE=0.07308968
    Iteracija 5/60: validaciona CDF-MSE=0.08018424
    Iteracija 10/60: validaciona CDF-MSE=0.07709665
  Izabrano: 1 iteracija; validaciona CDF-MSE=0.07308968
  Učenje za nezavisnu test proveru:
    Iteracija 1/1
    Prihvaćenih koraka: 1
Test: zajednička marginalna CDF-MSE=0.08486609
Test: prosečan broj pogodaka=1.2188
Test: raspodela pogodaka: 0 → 13, 1 → 27, 2 → 21, 3 → 3, 4 → 0, 5 → 0, 6 → 0, 7 → 0
Dijagnostika prethodnog kola: CDF-MSE=0.14432566; prosečni pogoci=1.3125
  Završno učenje: svih 2973 istorijskih prelaza
    Iteracija 1/1
    Prihvaćenih koraka: 1
NEXT: 9, 13, 29, 31, 34, 37, 38
Log-masa NEXT unutar naučene zajedničke raspodele: -8.94821625
Dijagnostika: najduži uzastopni niz=2; brojeva 1–19=2; brojeva 20–39=5
Marginalni modovi pozicija, samo dijagnostika:
  1, 4, 13, 13, 34, 37, 38
Ista kombinacija u istoriji: 0



======================================================================
loto7_4704_k83_loto_plus_1730.csv
Kola: 1730
SHA256: fc57041695a80b3c4e88c747acabf06e40b83a4914e21669bef89bcf469eef52
Kontrola: prethodno izvučeno kolo: 7, 10, 13, 27, 28, 31, 38
Prelazi: trening 1601, validacija 64, test 64
  Qiskit istorija: 1000/1730
  Provere prošle: Qiskit izlaz i gradijent.
  Izbor trajanja učenja:
    Iteracija 1/60: validaciona CDF-MSE=0.08495962
    Iteracija 5/60: validaciona CDF-MSE=0.08479278
    Iteracija 10/60: validaciona CDF-MSE=0.08138420
    Iteracija 15/60: validaciona CDF-MSE=0.08143154
    Iteracija 20/60: validaciona CDF-MSE=0.08005283
    Iteracija 25/60: validaciona CDF-MSE=0.08090260
  Izabrano: 19 iteracija; validaciona CDF-MSE=0.07991696
  Učenje za nezavisnu test proveru:
    Iteracija 1/19
    Iteracija 5/19
    Iteracija 10/19
    Iteracija 15/19
    Prihvaćenih koraka: 19
Test: zajednička marginalna CDF-MSE=0.08239908
Test: prosečan broj pogodaka=1.1094
Test: raspodela pogodaka: 0 → 17, 1 → 27, 2 → 16, 3 → 4, 4 → 0, 5 → 0, 6 → 0, 7 → 0
Dijagnostika prethodnog kola: CDF-MSE=0.14873120; prosečni pogoci=1.2812
  Završno učenje: svih 1729 istorijskih prelaza
    Iteracija 1/19
    Iteracija 5/19
    Iteracija 10/19
    Iteracija 15/19
    Prihvaćenih koraka: 19
NEXT: 3, 10, 19, 20, 24, 31, 38
Log-masa NEXT unutar naučene zajedničke raspodele: -7.93142923
Dijagnostika: najduži uzastopni niz=2; brojeva 1–19=3; brojeva 20–39=4
Marginalni modovi pozicija, samo dijagnostika:
  3, 10, 19, 20, 24, 28, 38
Ista kombinacija u istoriji: 0


======================================================================
ZAVRŠNE NEXT KOMBINACIJE
loto7_4704_k83.csv: 2, 13, 21, 24, 27, 29, 31
loto7_4704_k83_loto_2974.csv: 9, 13, 29, 31, 34, 37, 38
loto7_4704_k83_loto_plus_1730.csv: 3, 10, 19, 20, 24, 31, 38
"""



"""
model raspodele cele kombinacije, tako da zavisnosti i normalizaciju uči zajednički

Zajednička normalizacija cele kombinacije i 126 učenih kvantnih uglova. 
Nema zabrane uzastopnih brojeva niti pomeranja rezultata. 
Kod pri pokretanju proverava kvantno kolo, gradijent i zajedničku raspodelu.
"""
