// Code generated by go generate; DO NOT EDIT
// This file is generated by go genrated at 2022-01-26 15:50:36.315715425 +0800 CST m=+0.003539472

package entity

import (
	"fmt"

	"github.com/chensanle/milvus-sdk-go/v2/internal/proto/schema"
)

// ColumnBinaryVector generated columns type for BinaryVector
type ColumnBinaryVector struct {
	name   string
	dim    int
	values [][]byte
}

// Name returns column name
func (c *ColumnBinaryVector) Name() string {
	return c.name
}

// Type returns column FieldType
func (c *ColumnBinaryVector) Type() FieldType {
	return FieldTypeBinaryVector
}

// Len returns column data length
func (c *ColumnBinaryVector) Len() int {
	return len(c.values)
}

// Dim returns vector dimension
func (c *ColumnBinaryVector) Dim() int {
	return c.dim
}

// AppendValue append value into column
func (c *ColumnBinaryVector) AppendValue(i interface{}) error {
	v, ok := i.([]byte)
	if !ok {
		return fmt.Errorf("invalid type, expected []byte, got %T", i)
	}
	c.values = append(c.values, v)

	return nil
}

// Data returns column data
func (c *ColumnBinaryVector) Data() [][]byte {
	return c.values
}

// FieldData return column data mapped to schema.FieldData
func (c *ColumnBinaryVector) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type:      schema.DataType_BinaryVector,
		FieldName: c.name,
	}

	data := make([]byte, 0, len(c.values)*c.dim)

	for _, vector := range c.values {
		data = append(data, vector...)
	}

	fd.Field = &schema.FieldData_Vectors{
		Vectors: &schema.VectorField{
			Dim: int64(c.dim),

			Data: &schema.VectorField_BinaryVector{
				BinaryVector: data,
			},
		},
	}
	return fd
}

// NewColumnBinaryVector auto generated constructor
func NewColumnBinaryVector(name string, dim int, values [][]byte) *ColumnBinaryVector {
	return &ColumnBinaryVector{
		name:   name,
		dim:    dim,
		values: values,
	}
}

// ColumnFloatVector generated columns type for FloatVector
type ColumnFloatVector struct {
	name   string
	dim    int
	values [][]float32
}

// Name returns column name
func (c *ColumnFloatVector) Name() string {
	return c.name
}

// Type returns column FieldType
func (c *ColumnFloatVector) Type() FieldType {
	return FieldTypeFloatVector
}

// Len returns column data length
func (c *ColumnFloatVector) Len() int {
	return len(c.values)
}

// Dim returns vector dimension
func (c *ColumnFloatVector) Dim() int {
	return c.dim
}

// AppendValue append value into column
func (c *ColumnFloatVector) AppendValue(i interface{}) error {
	v, ok := i.([]float32)
	if !ok {
		return fmt.Errorf("invalid type, expected []float32, got %T", i)
	}
	c.values = append(c.values, v)

	return nil
}

// Data returns column data
func (c *ColumnFloatVector) Data() [][]float32 {
	return c.values
}

// FieldData return column data mapped to schema.FieldData
func (c *ColumnFloatVector) FieldData() *schema.FieldData {
	fd := &schema.FieldData{
		Type:      schema.DataType_FloatVector,
		FieldName: c.name,
	}

	data := make([]float32, 0, len(c.values)*c.dim)

	for _, vector := range c.values {
		data = append(data, vector...)
	}

	fd.Field = &schema.FieldData_Vectors{
		Vectors: &schema.VectorField{
			Dim: int64(c.dim),

			Data: &schema.VectorField_FloatVector{
				FloatVector: &schema.FloatArray{
					Data: data,
				},
			},
		},
	}
	return fd
}

// NewColumnFloatVector auto generated constructor
func NewColumnFloatVector(name string, dim int, values [][]float32) *ColumnFloatVector {
	return &ColumnFloatVector{
		name:   name,
		dim:    dim,
		values: values,
	}
}
