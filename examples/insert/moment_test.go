package main

import "testing"

func TestCreateMomentCollection(t *testing.T) {
	c := NewClient()

	CreateMomentCollection(c)
}

func TestCreateMomentIndex(t *testing.T) {
	c := NewClient()

	CreateMomentIndex(c)
}
