"""
Support functions for constructing and displaying model diagnostics.
"""
module diagnostics

export _isregularized, _nparams, _coef, _vcov, _stderror

using AxisArrays
using PrettyTables
import Base: show  # Overload for 1D and 2D AxisArrays

_isregularized(loglikelihood, loss) = loss != -loglikelihood  # loss == -LL + penalty

_nparams(prms...) = sum(length(p) for p in prms)

_coef(b::Matrix, rownames, colnames::Vector) = AxisArray(b, rownames=rownames, colnames=colnames)
_coef(b::Vector, rownames)                   = AxisArray(b, rownames=rownames)

# For coefficients that are vectors
function _stderror(vcov, rownames)
    ni, nj = size(vcov)
    @assert ni == nj
    @assert ni == length(rownames)
    data = [sqrt(abs(vcov[i,i])) for i = 1:ni]
    AxisArray(data, rownames=rownames)
end

# For coefficients that are matrices
function _stderror(vcov, rownames, colnames)
    ni, nj = length(rownames), length(colnames)
    nprms  = ni * nj
    @assert size(vcov, 1) == nprms
    @assert size(vcov, 2) == nprms
    se    = [sqrt(abs(vcov[i,i])) for i = 1:nprms]
    data  = reshape(se, ni, nj)
    AxisArray(data, rownames=rownames, colnames=colnames)
end

# For coefficients that are vectors
function _vcov(vcov, rownames)
    ni, nj = size(vcov)
    @assert ni == nj
    @assert ni == length(rownames)
    AxisArray(vcov, rownames=rownames, colnames=rownames)
end

# For coefficients that are matrices
function _vcov(vcov, xnames, ynames)
    ni, nj = length(xnames), length(ynames)
    nprms  = ni * nj
    @assert size(vcov, 1) == nprms
    @assert size(vcov, 2) == nprms
    colnames = construct_vcov_colnames(xnames, ynames)
    rownames = construct_vcov_rownames(xnames, ynames)
    AxisArray(vcov, rownames=rownames, colnames=colnames)
end

################################################################################
# Functions for displaying a fitted model

function construct_vcov_rownames(xnames, ynames)
    yname_maxlen  = maximum(length.(ynames)) + 4  # +2 for "y=_  "
    ynames_padded = rpad.(["y=$(yname)  " for yname in ynames], yname_maxlen)
    xname_maxlen  = maximum(length.(xnames)) + 2   # +2 for "x="
    xnames_padded = rpad.(["x=$(xname)" for xname in xnames], xname_maxlen)
    nprms         = length(ynames_padded) * length(xnames_padded)
    reshape(["$(yname)$(xname)" for xname in xnames_padded, yname in ynames_padded], nprms)
end

function construct_vcov_colnames(xnames, ynames)
    nprms = length(ynames) * length(xnames)
    reshape(["y=$(ylevel)  x=$(xname)" for xname in xnames, ylevel in ynames], nprms)
end

function Base.show(io::IO, ::MIME"text/plain", table::AxisArray{T,1,D,Tuple{A}}) where {T,D,A}
    pretty_table(table.data, header=[""], row_names=collect(table.axes[1].val))
end

function Base.show(io::IO, ::MIME"text/plain", table::AxisArray{T,2,D,Tuple{A1,A2}}) where {T,D,A1,A2}
    pretty_table(table.data, header=collect(table.axes[2].val), row_names=collect(table.axes[1].val))
end

end