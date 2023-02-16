import aerosandbox as asb

##### Section: Initialize Optimization
opti = asb.Opti(
    freeze_style='float',
    # variable_categories_to_freeze='design'
)
des = dict(category="design")
tra = dict(category="trajectory")

