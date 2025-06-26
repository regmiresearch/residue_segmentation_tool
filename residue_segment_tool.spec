# residue_segment_tool.spec

block_cipher = None

a = Analysis(
    ['residue_segment_tool.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('sam_vit_b.pth', '.'),  # include the model
        ('segment_anything/**/*', 'segment_anything'),  # include all SAM code
    ],
    hiddenimports=[
        'segment_anything',
        'segment_anything.build_sam',
        'segment_anything.predictor',
        'segment_anything.automatic_mask_generator',
        'segment_anything.utils',
        'segment_anything.modeling',
        'segment_anything.modeling.image_encoder',
        'segment_anything.modeling.mask_decoder',
        'segment_anything.modeling.prompt_encoder',
        'segment_anything.modeling.transformer',
        'segment_anything.modeling.sam',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ResidueSegmentationTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # set to False if you don't want a terminal to show
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ResidueSegmentationTool',
)
