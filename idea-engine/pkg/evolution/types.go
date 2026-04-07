// Package evolution implements genetic algorithm operators for the Idea
// Adaptation Engine (IAE). The genome encodes LARSA trading strategy
// parameters as a float64 slice. All operators treat the genome as an
// immutable value -- mutations and crossovers always return new copies.
package evolution

// Genome is the core encoding type: a slice of float64 parameters that
// represent a single LARSA strategy configuration.
type Genome []float64

// Clone returns a deep copy of the genome.
func (g Genome) Clone() Genome {
	c := make(Genome, len(g))
	copy(c, g)
	return c
}

// Individual pairs a Genome with its evaluated fitness result.
type Individual struct {
	// ID is a unique identifier assigned at creation.
	ID string
	// Genes is the parameter vector.
	Genes Genome
	// Fitness holds the most recent evaluation result.
	Fitness FitnessResult
	// Evaluated is true once Fitness has been populated.
	Evaluated bool
	// Generation is which generation this individual was created in.
	Generation int
	// ParentIDs contains the IDs of the parents (empty for gen 0).
	ParentIDs []string
}
