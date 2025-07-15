"""Add main docstring discription

"""

import logging
import itertools

import numpy as np
import pandas as pd

from rdkit import Chem

from SMILESX import utils

def augmentation(data_smiles, indices, data_extra=None, data_prop=None, check_smiles=True, augment=False, shuffle=False):
    """Augmentation

    Parameters
    ----------
    data_smiles:
        SMILES array for augmentation
    indices:
        Indices of the SMILES array for augmentation
    data_extra:
        Corresponding extra data array for augmentation
    data_prop:
        Corresponding property array for augmentation (default: None)
    check_smiles: bool
        Whether to verify SMILES correctness via RDKit (default: True)
    augment: bool
        Whether to augment the data by atom rotation (default: False)
    shuffle: bool
        Whether to generate all permutations of the input SMILES list (default: False)  

    Returns
    -------
    smiles_enum
        Array of augmented SMILES
    extra_enum
        Array of related additional inputs
    prop_enum
        Array of related property inputs
    miles_enum_card    
        Number of augmentation per SMILES
    """

    # Get the logger
    logger = logging.getLogger()
    
    if augment and not check_smiles:
        logging.error("ERROR:")
        logging.error("Augmentation is requested, but SMILES checking via RDKit is set to False.")
        logging.error("Augmentation cannot be performed on potentially invalid SMILES.")
        logging.error("")
        logging.error("*** AUGMENTATION ABORTED ***")
        raise utils.StopExecution

    smiles_enum = []
    prop_enum = []
    prop_clean = []
    extra_enum = []
    smiles_enum_card = []
    rejected_smiles = []
    indices_to_remove = []
    
    for csmiles, ismiles in enumerate(data_smiles.tolist()):
        if augment:
            enumerated_smiles = generate_smiles(ismiles, rotate=True, shuffle=shuffle)
        else:
            if check_smiles:
                enumerated_smiles = generate_smiles(ismiles, rotate=False, shuffle=shuffle)
            else:
                if not isinstance(ismiles, list):
                    ismiles = [ismiles]
                enumerated_smiles = [ismiles]
        if any(None in s for s in enumerated_smiles):
            rejected_smiles.extend(ismiles)
            indices_to_remove.append(csmiles)
        else:
            # Store indices where same index corresponds to the same orginal SMILES
            smiles_enum_card.extend([csmiles] * len(enumerated_smiles))
            smiles_enum.extend(enumerated_smiles)
            if data_prop is not None:
                prop_enum.extend([data_prop[csmiles]] * len(enumerated_smiles))
                prop_clean.extend(data_prop[csmiles])
            if data_extra is not None:
                extra_enum.extend([data_extra[csmiles]] * len(enumerated_smiles))
    if len(smiles_enum) == 0:
        logging.error("None of the provided SMILES is recognized as correct by RDKit.")
        logging.error("In case the SMILES data cannot be put to a correct format, set `check_smiles=False`.")
        logging.error("*Note: setting `check_smiles` to False disables augmentation.")
        logging.error("")
        logging.error("*** Process of inference is automatically aborted! ***")
        raise utils.StopExecution
    if len(rejected_smiles) > 0:
        logging.error("Some of the provided SMILES are recognized as incorrect by RDKit.")
        logging.error("The following list of SMILES have been rejected:")
        logging.error(rejected_smiles)
        logging.error("")

    if data_extra is None:
        extra_enum = None
    else:
        extra_enum = np.array(extra_enum)
    if data_prop is None:
        prop_enum = None
        prop_clean = None
    else:
        prop_enum = np.array(prop_enum)
        prop_clean = np.array(prop_clean)
    indices_clean = np.delete(indices, indices_to_remove)
    return smiles_enum, extra_enum, prop_enum, prop_clean, smiles_enum_card, indices_clean
##

def rotate_atoms(li, x):
    """Rotate atoms' index in a list.

    Parameters
    ----------
    li: list
        List to be rotated.
    x: int
        Index to be placed first in the list.

    Returns
    -------
        A list of rotated atoms.
    """

    return (li[x%len(li):]+li[:x%len(li)])
##

# Changed kekule from False to True to prevent aromatic notation
def generate_smiles(smiles, kekule=True, rotate=False, shuffle=False):
    """
    Generate SMILES list with optional augmentation (atom rotation)
    and shuffling of SMILES positions in a multi-SMILES input.

    Parameters
    ----------
    smiles : str or list(str)
        SMILES string or list of SMILES strings.
    kekule : bool
        Kekulize option setup.
    rotate : bool
        Rotation of atom indices for augmentation.
    shuffle : bool
        Whether to generate all permutations of the input SMILES list.

    Returns
    -------
    output_augm : list of lists
        Augmented (and/or shuffled) SMILES lists.
    """
    if isinstance(smiles, str):
        smiles = [smiles]


    original_smiles_list = [smiles]  # default: no shuffle

    if shuffle:
        perms = list(itertools.permutations(smiles))
        original_smiles_list = [list(p) for p in perms]
    
    all_augmented_lists = []

    for smi_list in original_smiles_list:
        augms = []
        for ismiles in smi_list:
            ismiles_augm = []
            if ismiles != '':
                mols = []
                try:
                    mol = Chem.MolFromSmiles(ismiles)
                    mols.append(mol)
                    n_atoms = mol.GetNumAtoms()
                    n_atoms_list = list(range(n_atoms))
                    if rotate and n_atoms != 0:
                        for i in range(n_atoms):
                            rot_mol = Chem.RenumberAtoms(mol, rotate_atoms(n_atoms_list, i))
                            mols.append(rot_mol)
                except:
                    mol = None

                for mol in mols:
                    try:
                        aug = Chem.MolToSmiles(
                            mol,
                            isomericSmiles=True,
                            kekuleSmiles=kekule,
                            rootedAtAtom=-1,
                            canonical=False,
                            allBondsExplicit=False,
                            allHsExplicit=False,
                        )
                        ismiles_augm.append(aug)
                    except:
                        ismiles_augm.append(None)

                # Remove duplicates
                ismiles_augm = list(dict.fromkeys(ismiles_augm))
                augms.append(ismiles_augm)
            else:
                augms.append([''])

        # Cartesian product of all augmentations per position
        combo = [list(au) for au in itertools.product(*augms)]
        all_augmented_lists.extend(combo)

    return all_augmented_lists
##