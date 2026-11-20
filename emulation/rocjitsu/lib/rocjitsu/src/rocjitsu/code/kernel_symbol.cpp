// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_symbol.h"

#include "rocjitsu/code/amdgpu_elf.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <memory>
#include <span>
#include <string_view>

namespace rocjitsu {
namespace {

using Image = std::span<const uint8_t>;

bool contains(Image image, uint64_t offset, uint64_t bytes) {
  return offset <= image.size() && bytes <= image.size() - offset;
}

template <typename T> T read_record(Image image, uint64_t offset) {
  T value;
  std::memcpy(&value, image.data() + offset, sizeof(value));
  return value;
}

bool is_elf(Image image) {
  return image.size() >= EI_MAGIC_SIZE && std::memcmp(image.data(), EI_MAGIC, EI_MAGIC_SIZE) == 0;
}

// ROCr copies PT_LOAD segments to their virtual offsets. The dynamic
// symbol table therefore remains available without the ELF section headers.
std::string find_dynamic_kernel_symbol(Image image, const Elf64_Phdr &segment,
                                       uint64_t descriptor_offset) {
  if (!contains(image, segment.p_vaddr, segment.p_memsz))
    return {};

  uint64_t symtab = 0, strtab = 0, strsz = 0, hash = 0;
  uint64_t syment = sizeof(Elf64_Sym);
  for (uint64_t i = 0; i < segment.p_memsz / sizeof(Elf64_Dyn); ++i) {
    const auto dyn = read_record<Elf64_Dyn>(image, segment.p_vaddr + i * sizeof(Elf64_Dyn));
    if (dyn.d_tag == DT_NULL)
      break;
    if (dyn.d_tag == DT_SYMTAB)
      symtab = dyn.d_un.d_val;
    else if (dyn.d_tag == DT_STRTAB)
      strtab = dyn.d_un.d_val;
    else if (dyn.d_tag == DT_STRSZ)
      strsz = dyn.d_un.d_val;
    else if (dyn.d_tag == DT_SYMENT)
      syment = dyn.d_un.d_val;
    else if (dyn.d_tag == DT_HASH)
      hash = dyn.d_un.d_val;
  }

  if (!symtab || !strtab || !hash || !strsz || syment != sizeof(Elf64_Sym) ||
      !contains(image, hash, 2 * sizeof(uint32_t)) || !contains(image, strtab, strsz))
    return {};
  const auto nsyms = read_record<uint32_t>(image, hash + sizeof(uint32_t));
  if (!contains(image, symtab, uint64_t{nsyms} * sizeof(Elf64_Sym)))
    return {};

  for (uint32_t i = 0; i < nsyms; ++i) {
    const auto sym = read_record<Elf64_Sym>(image, symtab + uint64_t{i} * sizeof(Elf64_Sym));
    if (sym.st_value != descriptor_offset || sym.st_shndx == SHN_UNDEF ||
        (sym.st_info & 0xf) != kElfSymbolTypeObject || sym.st_name >= strsz)
      continue;
    auto *name = reinterpret_cast<const char *>(image.data() + strtab + sym.st_name);
    const auto capacity = strsz - sym.st_name;
    const auto length = strnlen(name, capacity);
    const std::string_view symbol(name, length);
    if (length < capacity && length > 3 && symbol.ends_with(".kd"))
      return std::string(symbol.substr(0, length - 3));
  }
  return {};
}

// Kept for simdojo reports that historically accepted an inferred symbol.
// A note does not establish which descriptor belongs to a particular name.
std::string infer_kernel_symbol(Image image, const Elf64_Phdr &segment) {
  if (segment.p_filesz < sizeof(Elf64_Nhdr) || !contains(image, segment.p_offset, segment.p_filesz))
    return {};
  const auto header = read_record<Elf64_Nhdr>(image, segment.p_offset);
  if (header.n_type != NT_AMDGPU_METADATA)
    return {};

  const uint64_t name_aligned = (uint64_t{header.n_namesz} + 3) & ~uint64_t{3};
  if (name_aligned > segment.p_filesz - sizeof(header) ||
      header.n_descsz > segment.p_filesz - sizeof(header) - name_aligned)
    return {};
  const auto note =
      image.subspan(segment.p_offset + sizeof(header) + name_aligned, header.n_descsz);
  for (size_t pos = 2; pos + 2 < note.size(); ++pos) {
    if (note[pos] != '.' || note[pos + 1] != 'k' || note[pos + 2] != 'd')
      continue;
    const size_t end = pos + 3;
    if (end < note.size() && note[end] >= 0x20 && note[end] < 0x7f)
      continue;
    size_t start = pos;
    while (start > 0 && note[start - 1] >= 0x20 && note[start - 1] < 0x7f)
      --start;
    if (start != pos)
      return std::string(reinterpret_cast<const char *>(note.data() + start), pos - start);
  }
  return {};
}

std::string lookup_kernel_symbol(Image image, uint64_t descriptor_offset, bool infer) {
  if (!is_elf(image) || image.size() < sizeof(Elf64_Ehdr) || descriptor_offset >= image.size())
    return {};
  const auto header = read_record<Elf64_Ehdr>(image, 0);
  if (header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_machine != EM_AMDGPU || header.e_type != ET_DYN || !header.e_phoff ||
      header.e_phentsize != sizeof(Elf64_Phdr) || header.e_phnum > 64 ||
      !contains(image, header.e_phoff, uint64_t{header.e_phnum} * sizeof(Elf64_Phdr)))
    return {};

  auto segment = [&](uint16_t i) {
    return read_record<Elf64_Phdr>(image, header.e_phoff + uint64_t{i} * sizeof(Elf64_Phdr));
  };
  for (uint16_t i = 0; i < header.e_phnum; ++i) {
    const auto phdr = segment(i);
    if (phdr.p_type == PT_DYNAMIC) {
      auto symbol = find_dynamic_kernel_symbol(image, phdr, descriptor_offset);
      if (!symbol.empty())
        return symbol;
      break;
    }
  }
  if (infer) {
    for (uint16_t i = 0; i < header.e_phnum; ++i) {
      const auto phdr = segment(i);
      if (phdr.p_type == PT_NOTE) {
        auto symbol = infer_kernel_symbol(image, phdr);
        if (!symbol.empty())
          return symbol;
      }
    }
  }
  return {};
}

size_t find_outer_argument_list(std::string_view text) {
  size_t argument_list = std::string_view::npos;
  int angle_depth = 0;
  int parenthesis_depth = 0;

  for (size_t pos = 0; pos < text.size(); ++pos) {
    switch (text[pos]) {
    case '<':
      if (parenthesis_depth == 0)
        ++angle_depth;
      break;
    case '>':
      if (angle_depth > 0 && parenthesis_depth == 0)
        --angle_depth;
      break;
    case '(':
      if (angle_depth == 0 && parenthesis_depth == 0)
        argument_list = pos;
      ++parenthesis_depth;
      break;
    case ')':
      if (parenthesis_depth > 0)
        --parenthesis_depth;
      break;
    default:
      break;
    }
  }

  return argument_list;
}

} // namespace

std::string find_kernel_symbol(const uint8_t *kernel_object_ptr, const uint8_t *elf_base,
                               uint64_t elf_accessible) {
  if (!kernel_object_ptr || !elf_base || kernel_object_ptr < elf_base)
    return {};
  return lookup_kernel_symbol({elf_base, elf_accessible}, kernel_object_ptr - elf_base, true);
}

std::string find_loaded_kernel_symbol(Image allocation, uint64_t descriptor_offset) {
  if (descriptor_offset >= allocation.size())
    return {};
  // ROCr aligns loaded code objects to at least 4 KiB. An allocation can
  // contain more than one object, so look for the nearest loaded ELF header.
  constexpr uint64_t page_size = 4096;
  for (uint64_t offset = descriptor_offset & ~(page_size - 1);; offset -= page_size) {
    const auto image = allocation.subspan(offset);
    if (is_elf(image))
      return lookup_kernel_symbol(image, descriptor_offset - offset, false);
    if (offset == 0)
      return {};
  }
}

std::string demangle_kernel_symbol(std::string_view symbol) {
  if (symbol.empty())
    return {};

  std::string symbol_string(symbol);
  if (!symbol_string.starts_with("_Z"))
    return symbol_string;

  int status = 0;
  std::unique_ptr<char, decltype(&std::free)> demangled(
      abi::__cxa_demangle(symbol_string.c_str(), nullptr, nullptr, &status), &std::free);
  if (status != 0 || demangled == nullptr)
    return symbol_string;

  return std::string(demangled.get());
}

std::string kernel_display_name(std::string_view symbol) {
  std::string result = demangle_kernel_symbol(symbol);
  size_t parenthesis_position = find_outer_argument_list(result);
  if (parenthesis_position != std::string::npos)
    result.resize(parenthesis_position);
  if (result.starts_with("void "))
    result.erase(0, 5);

  std::erase_if(result, [](unsigned char c) { return std::isspace(c); });
  return result;
}

} // namespace rocjitsu
