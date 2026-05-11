# Project plan for IDP binder design

## Current state of IDP binder design

![alt text](image.png)

The latest SOTA does this:

- RFDiffusion trained on masked heterocomplexes (2 different chains - one for "target", one for "binder"). 
    - A mask is applied to the target's residues of the interface.
    - Around 50% of the residues are noised.
    - Model predicts the clean binder residues' coords ($x_0^b$) and 50% of the target's residues coords ($x_0^t$), using noised coords $x_t^b, \, x_t^t$, time step $t$ and the target sequence $s$.
    - The model is trained only on the backbone residues.
- Inference:
    - They generate around 10K-50K designs during inference, 
    - Inverse fold them with ProteinMPNN, 
    - Filter with AF2 pLDDT, etc. to get the initial "hits"
    - Noise and denoise the target-binder complex again, but use a smaller amount of noising steps. This is done to refine the prediction
    - Filter with ProteinMPNN + AF2 again


## Diffraction data and its use for IDPs

The diffraction data has implicit information about the rotamer states of a protein, possibly about different conformations of a protein, water molecules, as well as some statistically significant information about flexible regions of IDRs, which is hard to interpret visually.

The goal is to use that information alongside the sequence and structure information to train a binder generative model for IDPs/IDRs. The main idea is the following - **ensemble modelling improves co-folding**. Benefits:

- Possiblity to implicitly model altlocs (alternative locations) in PDB -> have more data to train on -> better co-folding models.
- Modelling side-chain wiggles explicitly allows to "tune" for specificity, which is associated with hydrogen bonds formed between target side-chains and binder side-chains, target side-chains and binder backbone, binder side-chains and target backbone. Additionally desolvation penalty can be modelled with $\Delta SASA$ (solvent accessible surface area) computed over several rotamers, which is substantially more robust than when it's computed for a single conformation. 
- Diffraction data might have useful signal for identifying IDR conformations -> it's an additional conditioning signal during training.
- If we can model water molecules explicitly, this would give access to desolvation as well. 

### Step 1 - PoC 

- Run a JEPA-style SSL on voxelised diffraction data in the reciprocal space build with Miller indices. That is the baseline.
- Show we are able to do representation learning with this setup. This is already good, since we could potentially apply it to IDRs.
- Generate a dataset of altlocs in parallel, using [Phenix](https://phenix-online.org/documentation/reference/ensemble_refinement.html). 10-50K monomers?
- JEPA-based encoder done, altloc dataset done -> finetune Complexa to output several conformers (how?)
    Check how well Complexa can fold (in the process) 
    - Idea 1: freeze Complexa, add the JEPA-encoder as an additional input, fuse its embeddings to the pair features at different layers, add a few trainable Proteina blocks to the head of the model.
    - Idea 2: Use LoRA with Complexa, but bias LoRA matrices with the JEPA embeddings. 
    - Finetune on a simple folding task - sequence to conformers

- Use Gaussian splatting (NeRFs) instead of voxels. Conformer generation utilising weight symmetries of neural fields??? Can we do JEPA on weights??? Need more reading
- Representation learning with the voxels/NeRFs/both can be a paper on its own if we manage to show some good performance on downstream tasks, especially with IDRs. 

### Step 2