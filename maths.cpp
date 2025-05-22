/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// Backward Pass ///////////////////////////////////////////////////////////////////////

const double local_t= ($(t)-$(rev_t))/$(trial_t);
scalar alpha= exp(-DT/$(tau_m)); scalar beta= exp(-DT/$(tau_syn));
scalar gamma= $(tau_m)/($(tau_m)-$(tau_syn)); scalar A= 0.0;

// Initialise modified Gaussian curve parameters. 
float a = 0.4;
float b = 4.4;
float c = 0.2;

// Don't calculate the intermediary variable A if this is the first trial.
if ($(trial) > 0) {

/////////////////////////////////////////////// Calculate Gaussian Weighting //////////////////////////////////////////////////////////////
    // For backwards pass, time is reversed.
    float   t_reverse   = 1 - local_t
    // Time squared.
    float   t_reverse_2 = t_reverse * t_reverse 
    // Variable c squared.
    float   c_2         =  c * c
    // Calculate the term that is exponentiated.
    float   exponential = exp( - (t_reverse_2) / (2 * c_2) )
    // Calculate weight_gaus, which is the Gaussian weighting for this given timestep, t_reverse.
    float   weight_gaus = ( a + b * t_reverse * exponential )
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Calculate intermediate variable A.
    // Different form if neuron ID == correct label. 
    if ($(id) == $(label)[($(trial)-1)*(int)$(N_batch)+$(batch)]) {
        A = 2 * weight_gaus * (1.0-$(SoftmaxVal)) / $(tau_m) / $(trial_t) / $(N_batch); 
    } else {
        A = -2 * weight_gaus * $(SoftmaxVal) / $(tau_m) / $(trial_t) / $(N_batch);  
    }
}
// Use A to update the adjoint variable lambda_I.
if (abs($(tau_m)-$(tau_syn)) < 1e-9) {
    $(lambda_I) = A + ( DT / $(tau_syn) * ( $(lambda_V) - A ) + ( $(lambda_I) - A) )*beta;
} else {
    $(lambda_I) = A + ( $(lambda_I) - A ) * beta + gamma * ( $(lambda_V) - A ) * (alpha-beta);
}

// Use A to update the adjoint variable lambda_V.
$(lambda_V) = A + ($(lambda_V)-A)*alpha;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// Forward Pass //////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////// Calculate Gaussian Weighting //////////////////////////////////////////////////////////
// Time squared.
float   time_2      = local_t * local_t 
// Variable c squared.
float   c_2         =  c * c
// Calculate the term that is exponentiated.
float   exponential = exp( - (time_2) / (2 * c_2) )
// Calculate weight_gaus, which is the Gaussian weighting for this given timestep, t_reverse.
float   weight_gaus = ( a + b * local_t * exponential )
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Update voltage sum.
$(sum_V) += 2 * weight_gaus * $(V) / $(trial_t) * DT;  

// Update voltage based on voltage decay equations. 
// Check if tau_m == tau_syn to avoid division by zero. 
if (abs($(tau_m)-$(tau_syn)) < 1e-9) {
    $(V)    = (DT/$(tau_m)*$(Isyn)+$(V))*exp(-DT/$(tau_m));
} else {
    $(V)    = $(tau_syn)/($(tau_m)-$(tau_syn))*$(Isyn)*(exp(-DT/$(tau_m))-exp(-DT/$(tau_syn)))+$(V)*exp(-DT/$(tau_m));
}



