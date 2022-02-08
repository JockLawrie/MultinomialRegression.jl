module ptables

export PTable

using PrettyTables
import Base.show  # To be overloaded
import Base.getindex  # To be overloaded

const ColumnTable = PrettyTables.ColumnTable

"""
A table that can be:
1. Indexed by row and column names.
2. Pretty printed.

The name PTable is a nod to the PrettyTables package.
"""
struct PTable{D, H, R}
    data::D
    header::H
    rownames::Vector{String}
    colname2index::Dict{String, Int}
    rowname2index::Dict{R, Int}  # primarykey => i. Tuple key type allows for multi-column primary keys.
end

function PTable(data, header, rownames, colnames)
    colname2index = Dict(colname => j for (j, colname) in enumerate(colnames))
    rowname2index = Dict(rowname => i for (i, rowname) in enumerate(rownames))
    PTable(data, header, rownames, colname2index, rowname2index)
end

show(io::IO, table::PTable) = pretty_table(table.data, header=table.header, row_names=table.rownames)

getindex(table::PTable, i, j)           = table.data[table.rowname2index[i], table.colname2index[j]]
getindex(table::PTable, i, j::Colon)    = table.data[table.rowname2index[i], :]
getindex(table::PTable, i::Colon, j)    = table.data[:, table.colname2index[j]]
getindex(table::PTable, i::Int, j::Int) = table.data[i, j]


end