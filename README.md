## Welcome to a Bayesian Modeling Framework

Running models is as simple as defining your MFF, metadata, and variable definitions. It is also as flexible as you need in order to solve a number of complex tasks.

### Usage

A model folder should be formatted as follows.
- model_directory
  - data 
    - data.csv (your mff)
    - metadata.json (not necessary but useful)
  - model_def.json

#### Metadata.json
```json
{
  "metadata": {
    // if other geos are detected in the MFF validation will fail 
    // list of geographies allowed in MFF
    "allowed_geos": [ 
      "KR",
      "AU",
      "US",
      "UK",
      "CA",
      "JP",
      "DE",
      "MX",
      "FR",
      "BR"
    ],
    // list of products allowed in the MFF
    "allowed_products": [ 
      "Total"
    ], 
    // list of outlets allowed in the MFF
    "allowed_outlets": [ 
      "Total"
    ],
    // list of allowed campaigns in the MFF
    "allowed_campaigns": [
      "Total"
    ],
    // list of allowed creatives in the MFF
    "allowed_creatives": [
      "Total"
    ],
    // list of variables that are required in the MFF
    "necessary_variables": null,
    
    "start_period": null, // currently not used
    "end_period": null, // currently not used
    /* 
    Allowed periodicities are Daily and Weekly.
    Weekly periodicity checks to ensure all dates are aligned to same week day.
     */
    "periodicity": "Weekly", 
    // Row ids define columns in the MFF that define unique records. Typically at least Period
    "row_ids": [
      "Geography",
      "Product",
      "Period"
    ]
  }
}
```

#### model_def.json
```json
{
  // variable_details expects a list of variables
  "variable_details": [
    // Variables are dictionaries
    // Variables need at least variable_name and variable_type 
    {
      "variable_name": "Placeholder Change Me", //REQUIRED: Name of variable in analytic dataframe
      "variable_type": "control", //REQUIRED: Options include [control, exog, base, media, none]
      /*
      deterministic_transform is not required
      defines function with known parameters not learned during model fitting.
      */
      "deterministic_transform": { 
        "functional_form": "linear", //Type of transform
        "params": null // Parameters required for transform
      },
      /* Only none and global standardize work, I highly recommend using global 
      standaredizeation for most indep vars outside of media. 
      That way intercept repressents the average rate when there is no media spend 
      and increases numerical stability. */
      "normalization": "none", 
      

      "time_transform": null, // Not implemented
      // Prior for coefficient 
      "coeff_prior": {
        "coeff_dist": "Normal", // Distribution for coefficient. The normal distribution is a reasonable choice for most controls
        "coef_params": {
          "mu": 0.0,
          "sigma": 3.0
        }
      },
      /* 
      Use fixed ind coeff if you want to learn a seperate coefficient without pooling.
      Use random coeff dims if you want to learn seperate coefficients with pooling
      Leaving both null or not defined will default to a single coefficient
      */
      "fixed_ind_coeff_dims": null,
      "random_coeff_dims": null
    },
    {
      "variable_name": "Media Placeholder Change Me",
      /*
      Media variables have a few differences from control variables 
      */
      "variable_type": "media",
      "deterministic_transform": {
        "functional_form": "linear",
        "params": null
      },
      "normalization": "none",
      "time_transform": null,
      "adstock": "delayed",
      "media_transform": "hill",
      "coeff_prior": {
        "coeff_dist": "LogNormal",
        "coeff_params": {
          "mu": -2.995732273553991,
          "sigma": 0.26236426446749106
        }
      },
      "fixed_ind_coeff_dims": null,
      "random_coeff_dims": null,
      "media_transform_prior": {
        "type": "Hill",
        "K_ave": 0.85,
        "K_std": 0.6,
        "n_ave": 1.5,
        "n_std": 1.2
      }
    }
  ],
  "artifact": "new_model",
  "fitted": false,
  "VOF": null
}
```


